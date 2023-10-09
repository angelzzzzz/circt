//===- VectorizeStates.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <variant>

#define DEBUG_TYPE "arc-vectorize-states"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_VECTORIZESTATES
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using llvm::SmallMapVector;
using llvm::SmallSetVector;

namespace {
struct TopologicalOrder {
  /// An integer rank assigned to each operation.
  DenseMap<Operation *, unsigned> opRanks;

  LogicalResult compute(Block *block);
  unsigned get(Operation *op) const {
    auto it = opRanks.find(op);
    assert(it != opRanks.end() && "op has no rank");
    return it->second;
  }
};
} // namespace

/// Assign each operation in the given block a topological rank. Stateful
/// elements are assigned rank 0. All other operations receive the maximum rank
/// of their users, plus one.
LogicalResult TopologicalOrder::compute(Block *block) {
  LLVM_DEBUG(llvm::dbgs() << "- Computing topological order in block " << block
                          << "\n");
  struct WorklistItem {
    WorklistItem(Operation *op) : userIt(op->user_begin()) {}
    Operation::user_iterator userIt;
    unsigned rank = 0;
  };
  SmallMapVector<Operation *, WorklistItem, 16> worklist;
  for (auto &op : *block) {
    if (opRanks.contains(&op))
      continue;
    worklist.insert({&op, WorklistItem(&op)});
    while (!worklist.empty()) {
      auto &[op, item] = worklist.back();
      if (auto stateOp = dyn_cast<StateOp>(op)) {
        if (stateOp.getLatency() > 0)
          item.userIt = op->user_end();
      } else if (auto writeOp = dyn_cast<MemoryWritePortOp>(op)) {
        item.userIt = op->user_end();
      }
      if (item.userIt == op->user_end()) {
        opRanks.insert({op, item.rank});
        worklist.pop_back();
        continue;
      }
      if (auto rankIt = opRanks.find(*item.userIt); rankIt != opRanks.end()) {
        item.rank = std::max(item.rank, rankIt->second + 1);
        ++item.userIt;
        continue;
      }
      if (!worklist.insert({*item.userIt, WorklistItem(*item.userIt)}).second)
        return op->emitError("dependency cycle");
    }
  }
  return success();
}

namespace {
struct StateFingerprint {
  StringAttr arcName;
  unsigned rank = 0;
  unsigned latency = 0;
  SmallVector<Value, 1> operands;
  SmallVector<std::optional<std::pair<OperationName, DictionaryAttr>>, 0>
      operandDefs;

  StateFingerprint(StringAttr arcName) : arcName(arcName) {}

  bool operator==(const StateFingerprint &other) const {
    return arcName == other.arcName && latency == other.latency &&
           operands == other.operands && operandDefs == other.operandDefs;
  }

  void addExactMatch(Value operand) {
    if (operand)
      operands.push_back(operand);
  }

  void addDefMatch(Value operand) {
    auto *defOp = operand.getDefiningOp();
    if (!defOp) {
      operandDefs.push_back({});
      return;
    }
    NamedAttrList attrs;
    for (auto attr : defOp->getAttrs())
      if (attr.getName() != "names" && attr.getName() != "name" &&
          attr.getName() != "value")
        attrs.push_back(attr);
    operandDefs.push_back(
        {{defOp->getName(), attrs.getDictionary(defOp->getContext())}});
  }
};
} // namespace

namespace llvm {
template <>
struct DenseMapInfo<StateFingerprint> {
  static inline StateFingerprint getEmptyKey() {
    return StateFingerprint(DenseMapInfo<StringAttr>::getEmptyKey());
  }
  static inline StateFingerprint getTombstoneKey() {
    return StateFingerprint(DenseMapInfo<StringAttr>::getTombstoneKey());
  }
  static unsigned getHashValue(const StateFingerprint &x) {
    return hash_value(x.arcName) ^ hash_value(x.latency) ^
           hash_value(ArrayRef(x.operands)) ^
           hash_value(ArrayRef(x.operandDefs));
  }
  static bool isEqual(const StateFingerprint &lhs,
                      const StateFingerprint &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace {
struct BlockVectorizer {
  Block *block;
  SymbolTable &symtbl;
  // DenseMap<StateFingerprint, SmallVector<StateOp, 1>> states;
  // DenseMap<llvm::hash_code, SmallVector<VectorPackOp, 0>> vectorPackOps;
  // SmallVector<VectorPackOp> vectorPackWorklist;
  TopologicalOrder order;

  BlockVectorizer(Block *block, SymbolTable &symtbl)
      : block(block), symtbl(symtbl) {}
  LogicalResult vectorize();
  LogicalResult createInitialVectorizeOps();
  LogicalResult vectorizeArcs();
  LogicalResult vectorizeStates();
  LogicalResult propagate();

  // VectorPackOp buildVector(ImplicitLocOpBuilder &builder, ValueRange
  // operands) {
  //   auto hash = llvm::hash_combine_range(operands.begin(), operands.end());
  //   auto &ops = vectorPackOps[hash];
  //   for (auto op : ops)
  //     if (op.getOperands() == operands)
  //       return op;
  //   auto op = builder.create<VectorPackOp>(
  //       hw::ArrayType::get(operands[0].getType(), operands.size()),
  //       operands);
  //   ops.push_back(op);
  //   vectorPackWorklist.push_back(op);
  //   return op;
  // }
};
} // namespace

LogicalResult BlockVectorizer::vectorize() {
  LLVM_DEBUG(llvm::dbgs() << "- Vectorizing block " << block << "\n");
  auto *context = block->getParent()->getContext();

  if (failed(order.compute(block)))
    return failure();
  // DEBUG: Annotate topo order.
  for (auto [op, rank] : order.opRanks)
    op->setAttr("rank", IntegerAttr::get(IntegerType::get(context, 64), rank));

  if (failed(createInitialVectorizeOps()))
    return failure();

  if (failed(vectorizeArcs()))
    return failure();

  return success();
}

LogicalResult BlockVectorizer::createInitialVectorizeOps() {
  // As a starting point, group lat>0 uses of the same arc that have the same
  // clock, enable, and reset.
  using Key = std::tuple<StringAttr, unsigned, Value, Value, Value>;
  DenseMap<Key, SmallVector<StateOp, 1>> stateGroups;
  unsigned numOps = 0;
  for (auto stateOp : block->getOps<StateOp>()) {
    if (stateOp.getLatency() == 0 || stateOp.getNumResults() != 1 ||
        !isa<IntegerType>(stateOp.getResult(0).getType()) ||
        llvm::any_of(stateOp.getOperandTypes(), [](auto type) {
          return !isa<IntegerType, seq::ClockType>(type);
        }))
      continue;
    Key key = std::make_tuple(stateOp.getArcAttr().getAttr(),
                              stateOp.getLatency(), stateOp.getClock(),
                              stateOp.getEnable(), stateOp.getReset());
    stateGroups[key].push_back(stateOp);
    ++numOps;
  }
  LLVM_DEBUG(llvm::dbgs() << "  - Found " << stateGroups.size()
                          << " groups among " << numOps << " states\n");

  // SmallVector<std::pair<unsigned, StateOp>> sorted;
  // for (auto &[key, ops] : stateGroups)
  //   sorted.emplace_back(ops.size(), ops[0]);
  // llvm::sort(sorted, [](auto &a, auto &b) { return a.first > b.first; });
  // LLVM_DEBUG(llvm::dbgs() << "  - Biggest groups:\n");
  // for (unsigned i = 0; i < 10 && i < sorted.size(); ++i)
  //   LLVM_DEBUG(llvm::dbgs() << "    - " << sorted[i].first
  //                           << " lanes: " << sorted[i].second << "\n");

  // From these groups, create the initial vectorization ops.
  unsigned numGroupsCreated = 0;
  unsigned numOpsRemoved = 0;
  SmallDenseMap<Value, Value> clockToI1Casts;
  for (auto &[key, stateOps] : stateGroups) {
    if (stateOps.size() < 2)
      continue;
    ImplicitLocOpBuilder builder(stateOps[0].getLoc(), stateOps[0]);

    // Package the N operands of each of the M ops into an array of N vectors of
    // M operands each.
    SmallVector<SmallVector<Value, 4>> vectorOperands;
    vectorOperands.resize(stateOps[0].getNumOperands());
    for (auto stateOp : stateOps) {
      assert(stateOp.getNumOperands() == vectorOperands.size());
      for (auto [into, operand] :
           llvm::zip(vectorOperands, stateOp.getOperands())) {
        if (isa<seq::ClockType>(operand.getType())) {
          auto &cast = clockToI1Casts[operand];
          if (!cast)
            cast = builder.create<seq::FromClockOp>(operand);
          operand = cast;
        }
        into.push_back(operand);
      }
    }
    SmallVector<ValueRange> vectorOperandRanges;
    vectorOperandRanges.assign(vectorOperands.begin(), vectorOperands.end());

    SmallVector<Type> vectorResultTypes(stateOps.size(),
                                        stateOps[0].getResult(0).getType());

    // Create the `VectorizeOp`.
    auto vectorizeOp =
        builder.create<VectorizeOp>(vectorResultTypes, vectorOperandRanges);
    auto &vectorizeBlock = vectorizeOp.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&vectorizeBlock);

    // Add block arguments.
    IRMapping argMapping;
    for (auto [castOperand, originalOperand] :
         llvm::zip(vectorOperands, stateOps[0].getOperands())) {
      Value arg = vectorizeBlock.addArgument(castOperand[0].getType(),
                                             originalOperand.getLoc());
      if (isa<seq::ClockType>(originalOperand.getType()))
        arg = builder.create<seq::ToClockOp>(arg);
      argMapping.map(originalOperand, arg);
    }
    // if (stateOps.size() > 500)
    //   LLVM_DEBUG(llvm::dbgs() << "  - Created " << vectorizeOp << "\n");

    // Clone operation into the block.
    auto *clonedOp = builder.clone(*stateOps[0], argMapping);

    // Create the return op.
    builder.create<VectorizeReturnOp>(clonedOp->getResult(0));

    // Replace uses of the original ops with the vectorized version.
    for (auto [stateOp, result] :
         llvm::zip(stateOps, vectorizeOp.getResults())) {
      stateOp.getResult(0).replaceAllUsesWith(result);
      stateOp.erase();
    }

    ++numGroupsCreated;
    numOpsRemoved += stateOps.size();
  }
  LLVM_DEBUG(llvm::dbgs() << "  - Moved " << numOpsRemoved << " ops into "
                          << numGroupsCreated << " vectorize ops\n");

  return success();
}

LogicalResult BlockVectorizer::vectorizeArcs() {
  auto *context = block->getParent()->getContext();

  DenseMap<std::pair<StringAttr, unsigned>, DefineOp> vectorizedArcs;

  SmallVector<VectorizeOp> worklist;
  auto ops = block->getOps<VectorizeOp>();
  worklist.assign(ops.begin(), ops.end());
  LLVM_DEBUG(llvm::dbgs() << "  - Processing " << worklist.size()
                          << " vectorize ops\n");

  while (!worklist.empty()) {
    auto vectorizeOp = worklist.pop_back_val();
    auto *bodyBlock = &vectorizeOp.getBody().front();
    unsigned numLanes = vectorizeOp.getResults().size();
    if (numLanes == 1)
      numLanes =
          vectorizeOp.getResults().front().getType().getIntOrFloatBitWidth() /
          bodyBlock->getTerminator()
              ->getOperand(0)
              .getType()
              .getIntOrFloatBitWidth();

    // Vectorize the block arguments.
    for (auto &arg : bodyBlock->getArguments()) {
      auto bw = arg.getType().getIntOrFloatBitWidth();
      bw *= numLanes;
      arg.setType(IntegerType::get(context, bw));
    }

    // Vectorize the operations.
    for (auto &op : llvm::make_early_inc_range(*bodyBlock)) {
      ImplicitLocOpBuilder builder(op.getLoc(), &op);
      if (isa<VectorizeReturnOp>(&op))
        continue;

      // Hacky lowering for clock casts, since we assume that they only feed the
      // arc's clock.
      if (auto toClockOp = dyn_cast<seq::ToClockOp>(&op)) {
        auto bit = builder.create<comb::ExtractOp>(toClockOp.getInput(), 0, 1);
        toClockOp.getInputMutable().assign(bit);
        continue;
      }

      // Vectorize `StateOp`s by vectorizing their results and creating a
      // vectorized version of the arc.
      if (auto stateOp = dyn_cast<StateOp>(&op)) {
        // Hacky lowering for the non-vectorizable enable and reset operands.
        auto singleBit = [&](MutableOperandRange operand) {
          auto bit = builder.create<comb::ExtractOp>(operand[0].get(), 0, 1);
          operand.assign(bit);
        };
        if (stateOp.getEnable())
          singleBit(stateOp.getEnableMutable());
        if (stateOp.getReset())
          singleBit(stateOp.getResetMutable());

        // Vectorize the result types.
        for (auto result : stateOp.getResults()) {
          auto bw = result.getType().getIntOrFloatBitWidth();
          bw *= numLanes;
          result.setType(IntegerType::get(context, bw));
        }
        continue;
      }

      if (op.getNumRegions() > 0)
        return op.emitOpError("cannot be trivially vectorized; has regions");

      // Trivial vectorization by replicating the op.
      LLVM_DEBUG(llvm::dbgs() << "  - Vectorizing " << op << "\n");
      SmallVector<SmallVector<Value>, 1> laneResults;
      laneResults.resize(op.getNumResults());
      for (unsigned laneIdx = 0; laneIdx < numLanes; ++laneIdx) {
        IRMapping mapping;
        for (auto operand : op.getOperands()) {
          if (isa<hw::ArrayType>(operand.getType())) {
            auto requiredWidth = llvm::Log2_64_Ceil(numLanes);
            auto idx = builder.create<hw::ConstantOp>(
                builder.getIntegerType(requiredWidth), laneIdx);
            auto lane = builder.create<hw::ArrayGetOp>(operand, idx);
            // LLVM_DEBUG(llvm::dbgs() << "    - Created " << lane << "\n");
            mapping.map(operand, lane);
          } else {
            auto bw = operand.getType().getIntOrFloatBitWidth() / numLanes;
            auto lane =
                builder.create<comb::ExtractOp>(operand, laneIdx * bw, bw);
            // LLVM_DEBUG(llvm::dbgs() << "    - Created " << lane << "\n");
            mapping.map(operand, lane);
          }
        }
        auto *cloned = builder.clone(op, mapping);
        // LLVM_DEBUG(llvm::dbgs() << "    - Created " << *cloned << "\n");
        for (auto [laneResult, opResult] :
             llvm::zip(laneResults, cloned->getResults()))
          laneResult.push_back(opResult);
      }
      for (auto [laneResult, opResult] :
           llvm::zip(laneResults, op.getResults())) {
        std::reverse(laneResult.begin(), laneResult.end());
        if (isa<hw::ArrayType>(opResult.getType())) {
          auto createOp = builder.create<hw::ArrayCreateOp>(laneResult);
          // LLVM_DEBUG(llvm::dbgs() << "    - Created " << createOp << "\n");
          opResult.replaceAllUsesWith(createOp);
        } else {
          auto concatOp = builder.create<comb::ConcatOp>(laneResult);
          // LLVM_DEBUG(llvm::dbgs() << "    - Created " << concatOp << "\n");
          opResult.replaceAllUsesWith(concatOp);
        }
      }
      assert(op.use_empty());
      op.erase();
    }

    // Vectorize the arc definitions.
    for (auto stateOp : bodyBlock->getOps<StateOp>()) {
      auto key = std::make_pair(stateOp.getArcAttr().getAttr(), numLanes);
      auto &vectorizedArc = vectorizedArcs[key];
      if (vectorizedArc) {
        stateOp.setArcAttr(FlatSymbolRefAttr::get(vectorizedArc));
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "  - Vectorizing " << stateOp.getArcAttr()
                              << " into " << numLanes << " lanes\n");

      auto scalarArc = symtbl.lookup<DefineOp>(stateOp.getArc());
      if (!scalarArc)
        return failure();

      // Create a new arc definition with vectorized arguments and results.
      OpBuilder builder(scalarArc);
      TypeRange argTypes = stateOp.getInputs().getTypes();
      TypeRange resultTypes = stateOp.getResultTypes();
      vectorizedArc = builder.create<DefineOp>(
          scalarArc.getLoc(),
          builder.getStringAttr(scalarArc.getSymName() + "_vec" +
                                Twine(numLanes)),
          builder.getFunctionType(argTypes, resultTypes));
      symtbl.insert(vectorizedArc);

      // Create the block arguments.
      auto &arcBlock = vectorizedArc.getBody().emplaceBlock();
      for (auto [type, arg] : llvm::zip(argTypes, scalarArc.getArguments())) {
        arcBlock.addArgument(type, arg.getLoc());
      }
      builder.setInsertionPointToStart(&arcBlock);

      // Create a vectorize op that will hold the entire arc body.
      SmallVector<ValueRange> vectorOperands(arcBlock.args_begin(),
                                             arcBlock.args_end());
      auto vectorizeOp = builder.create<VectorizeOp>(
          scalarArc.getLoc(), stateOp.getResult(0).getType(), vectorOperands);
      builder.create<arc::OutputOp>(scalarArc.getLoc(),
                                    vectorizeOp.getResults());

      // Create the block arguments for the vectorize op.
      auto &vectorizeBlock = vectorizeOp.getBody().emplaceBlock();
      IRMapping mapping;
      for (auto arg : scalarArc.getArguments())
        mapping.map(arg,
                    vectorizeBlock.addArgument(arg.getType(), arg.getLoc()));
      builder.setInsertionPointToStart(&vectorizeBlock);

      // Clone the operations from the scalar arc into the vectorize op.
      for (auto &op : scalarArc.getBodyBlock()) {
        if (auto outputOp = dyn_cast<arc::OutputOp>(&op)) {
          builder.create<VectorizeReturnOp>(
              outputOp.getLoc(), mapping.lookup(outputOp.getOutputs()[0]));
        } else {
          builder.clone(op, mapping);
        }
      }

      stateOp.setArcAttr(FlatSymbolRefAttr::get(vectorizedArc));
      worklist.push_back(vectorizeOp);
    }
  }

  return success();
}

LogicalResult BlockVectorizer::vectorizeStates() {
  // LLVM_DEBUG(llvm::dbgs() << "- Vectorizing " << states.size() << "
  // groups\n");

  // // Vectorize `StateOp`s that have siblings with the same ops feeding their
  // // operands.
  // unsigned numVectorized = 0;
  // unsigned numRemoved = 0;
  // for (auto &[fingerprint, stateOps] : states) {
  //   if (stateOps.size() < 2)
  //     continue;
  //   ++numVectorized;
  //   numRemoved += stateOps.size();
  //   LLVM_DEBUG(llvm::dbgs() << "  - Vectorizing " << stateOps.size() << " way
  //   "
  //                           << fingerprint.arcName << " (rank "
  //                           << fingerprint.rank << ")\n");

  //   auto protoOp = stateOps[0];
  //   ImplicitLocOpBuilder builder(protoOp.getLoc(), protoOp);

  //   // Vectorize each operand.
  //   SmallVector<Value> vectorizedOperands;
  //   for (unsigned opNum = 0; opNum < protoOp.getInputs().size(); ++opNum) {
  //     SmallVector<Value> args;
  //     for (auto stateOp : stateOps)
  //       args.push_back(stateOp.getInputs()[opNum]);
  //     // std::reverse(args.begin(), args.end());
  //     vectorizedOperands.push_back(buildVector(builder, args));
  //   }

  //   // Build the result types.
  //   SmallVector<Type> resultTypes;
  //   for (auto type : protoOp.getResultTypes())
  //     resultTypes.push_back(hw::ArrayType::get(type, stateOps.size()));

  //   // Create a vectorized version of the state op.
  //   auto parallelOp =
  //       builder.create<ParallelOp>(resultTypes, vectorizedOperands);
  //   auto &bodyBlock = parallelOp.getBody().emplaceBlock();
  //   SmallVector<Value> mappedInputs;
  //   for (auto input : protoOp.getInputs())
  //     mappedInputs.push_back(
  //         bodyBlock.addArgument(input.getType(), protoOp.getLoc()));
  //   builder.setInsertionPointToStart(&bodyBlock);
  //   auto innerStateOp = builder.create<StateOp>(
  //       protoOp.getArcAttr(), protoOp.getResultTypes(), protoOp.getClock(),
  //       protoOp.getEnable(), protoOp.getReset(), protoOp.getLatency(),
  //       mappedInputs);
  //   builder.create<arc::OutputOp>(innerStateOp.getOutputs());
  //   builder.setInsertionPoint(protoOp);

  //   // Devectorize each result.
  //   SmallVector<ValueRange> devectorizedResults;
  //   for (auto result : parallelOp.getResults()) {
  //     auto arrayType = cast<hw::ArrayType>(result.getType());
  //     SmallVector<Type> explodedResultTypes;
  //     for (unsigned i = 0; i < arrayType.getNumElements(); ++i)
  //       explodedResultTypes.push_back(arrayType.getElementType());
  //     devectorizedResults.push_back(
  //         builder.create<VectorUnpackOp>(explodedResultTypes, result)
  //             .getElements());
  //   }

  //   // Replace uses of the original states with elements of the vectorized
  //   // results.
  //   for (auto [index, stateOp] : llvm::enumerate(stateOps)) {
  //     for (auto [oldResult, newResult] :
  //          llvm::zip(stateOp.getResults(), devectorizedResults))
  //       oldResult.replaceAllUsesWith(newResult[index]);
  //     assert(stateOp.use_empty());
  //     stateOp.erase();
  //   }
  // }
  // LLVM_DEBUG(llvm::dbgs() << "  - Created " << numVectorized
  //                         << " vectorized states (" << numRemoved
  //                         << " scalar states removed)\n");
  // states.clear();

  // // Simplify array create ops.
  // unsigned readIdx = 0, writeIdx = 0, endIdx = vectorPackWorklist.size();
  // for (; readIdx < endIdx; ++readIdx) {
  //   auto createOp = vectorPackWorklist[readIdx];
  //   auto explodeOp =
  //   createOp.getOperand(0).getDefiningOp<hw::ArrayExplodeOp>(); if (explodeOp
  //   &&
  //       createOp.getOperands() == llvm::reverse(explodeOp.getResults())) {
  //     createOp.replaceAllUsesWith(explodeOp.getInput());
  //     createOp.erase();
  //     continue;
  //   }
  //   vectorPackWorklist[writeIdx++] = createOp;
  // }
  // vectorPackWorklist.resize(writeIdx);

  return success();
}

LogicalResult BlockVectorizer::propagate() {
  // LLVM_DEBUG(llvm::dbgs() << "- Growing parallel regions in " << block <<
  // "\n"); LLVM_DEBUG(llvm::dbgs() << "  - " << vectorPackWorklist.size()
  //                         << " arrays\n");

  // // Try the trivial ones first.
  // unsigned numStates = 0;
  // for (auto arrayCreateOp : vectorPackWorklist) {
  //   auto onlyUsedHere =
  //       llvm::all_of(arrayCreateOp.getOperands(), [&](Value operand) {
  //         auto *defOp = operand.getDefiningOp();
  //         if (!defOp)
  //           return false;
  //         for (auto *user : defOp->getUsers())
  //           if (user != arrayCreateOp)
  //             return false;
  //         return true;
  //       });
  //   if (!onlyUsedHere)
  //     continue;
  //   LLVM_DEBUG(llvm::dbgs() << "  - Found trivial " << arrayCreateOp <<
  //   "\n");

  //   // Mark states for vectorization.
  //   for (auto operand : llvm::reverse(arrayCreateOp.getOperands())) {
  //     auto stateOp = operand.getDefiningOp<StateOp>();
  //     if (!stateOp)
  //       continue;
  //     ++numStates;
  //     StateFingerprint fp(stateOp.getArcAttr().getAttr());
  //     fp.latency = stateOp.getLatency();
  //     fp.addExactMatch(stateOp.getClock());
  //     fp.addExactMatch(stateOp.getEnable());
  //     fp.addExactMatch(stateOp.getReset());
  //     auto &slot = states[fp];
  //     if (!llvm::is_contained(slot, stateOp))
  //       slot.push_back(stateOp);
  //   }
  // }
  // LLVM_DEBUG(llvm::dbgs() << "  - " << states.size() << " groups among "
  //                         << numStates << " states\n");
  // vectorPackWorklist.clear();

  return success();
}

namespace {
struct VectorizeStatesPass
    : public impl::VectorizeStatesBase<VectorizeStatesPass> {
  void runOnOperation() override;
};
} // namespace

void VectorizeStatesPass::runOnOperation() {
  auto &symtbl = getAnalysis<SymbolTable>();
  for (auto moduleOp : getOperation().getOps<hw::HWModuleOp>()) {
    LLVM_DEBUG(llvm::dbgs() << "Vectorizing states in "
                            << moduleOp.getModuleNameAttr() << "\n");
    auto result = moduleOp.walk([&](Block *block) {
      if (!mayHaveSSADominance(*block->getParent()))
        if (failed(BlockVectorizer(block, symtbl).vectorize()))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();
  }

  // // Apply rewrite patterns to simplify structure.
  // RewritePatternSet patterns(&getContext());
  // patterns.add<VectorPackUnpackPattern, TrivialParallelOpMergePattern>(
  //     &getContext());
  // if (failed(applyPatternsAndFoldGreedily(getOperation(),
  // std::move(patterns))))
  //   return signalPassFailure();

  // // Dump parallel ops that can be sunk into other parallel ops.
  // llvm::dbgs() << "Sinkable ParallelOps:\n";
  // LLVM_DEBUG({
  //   getOperation().walk([&](ParallelOp op) {
  //     ParallelOp sameUser;
  //     for (auto *user : op->getUsers()) {
  //       if (user == op)
  //         continue;
  //       if (auto parallelUser = dyn_cast<ParallelOp>(user)) {
  //         if (!sameUser)
  //           sameUser = parallelUser;
  //         else if (sameUser != parallelUser)
  //           return;
  //       } else {
  //         return;
  //       }
  //     }
  //     if (!sameUser)
  //       return;
  //     llvm::dbgs() << "- Can sink " << op << " into " << sameUser << "\n";
  //   });
  // });

  // return;

  // // Dump some parallelization cost information.
  // DenseMap<StringAttr, unsigned> arcCosts;
  // getOperation().getParentOp<mlir::ModuleOp>().walk([&](DefineOp op) {
  //   arcCosts[op.getSymNameAttr()] =
  //       std::distance(op.getBodyBlock().begin(), op.getBodyBlock().end());
  // });

  // SmallVector<std::tuple<ParallelOp, unsigned, unsigned>> costs;
  // getOperation().walk([&](ParallelOp parallelOp) {
  //   unsigned cost = 0;
  //   for (auto op : parallelOp.getBodyBlock().getOps<StateOp>())
  //     cost += arcCosts.lookup(op.getArcAttr().getAttr());
  //   unsigned ways =
  //       hw::type_cast<hw::ArrayType>(parallelOp.getOutputs()[0].getType())
  //           .getNumElements();
  //   costs.push_back({parallelOp, cost, ways});
  // });
  // llvm::sort(costs, [](auto &a, auto &b) {
  //   return std::get<1>(a) * std::get<2>(a) < std::get<1>(b) * std::get<2>(b);
  // });
  // LLVM_DEBUG({
  //   llvm::dbgs() << "- Parallelization costs:\n";
  //   for (auto [parallelOp, cost, ways] : costs) {
  //     llvm::dbgs() << "  - " << (cost * ways) << " (" << cost << " x " <<
  //     ways
  //                  << "): " << parallelOp << "\n";
  //   }
  // });
}

std::unique_ptr<Pass> arc::createVectorizeStatesPass() {
  return std::make_unique<VectorizeStatesPass>();
}
