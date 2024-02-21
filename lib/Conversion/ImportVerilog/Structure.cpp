//===- Structure.cpp - Slang hierarchy conversion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/Compilation.h"

// #include "slang/ast/ASTVisitor.h"
// #include "slang/ast/Symbol.h"
// #include "slang/ast/symbols/CompilationUnitSymbols.h"
// #include "slang/ast/symbols/InstanceSymbols.h"
// #include "slang/ast/symbols/VariableSymbols.h"
// #include "slang/ast/types/AllTypes.h"
// #include "slang/ast/types/Type.h"
// #include "slang/syntax/SyntaxVisitor.h"
// #include "llvm/ADT/StringRef.h"

using namespace circt;
using namespace ImportVerilog;

//===----------------------------------------------------------------------===//
// Top-Level Item Conversion
//===----------------------------------------------------------------------===//

namespace {
struct RootMemberVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  RootMemberVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  /// Skip semicolons.
  LogicalResult visit(const slang::ast::EmptyMemberSymbol &) {
    return success();
  }

  /// Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported top-level construct: ")
        << slang::ast::toString(node.kind);
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Module Member Conversion
//===----------------------------------------------------------------------===//

static moore::ProcedureKind
convertProcedureKind(slang::ast::ProceduralBlockKind kind) {
  switch (kind) {
  case slang::ast::ProceduralBlockKind::Always:
    return moore::ProcedureKind::Always;
  case slang::ast::ProceduralBlockKind::AlwaysComb:
    return moore::ProcedureKind::AlwaysComb;
  case slang::ast::ProceduralBlockKind::AlwaysLatch:
    return moore::ProcedureKind::AlwaysLatch;
  case slang::ast::ProceduralBlockKind::AlwaysFF:
    return moore::ProcedureKind::AlwaysFF;
  case slang::ast::ProceduralBlockKind::Initial:
    return moore::ProcedureKind::Initial;
  case slang::ast::ProceduralBlockKind::Final:
    return moore::ProcedureKind::Final;
  }
  llvm_unreachable("all procedure kinds handled");
}

namespace {
struct MemberVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  MemberVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  /// Skip semicolons.
  LogicalResult visit(const slang::ast::EmptyMemberSymbol &) {
    return success();
  }

  // Skip parameters. The AST is already monomorphized.
  LogicalResult visit(const slang::ast::ParameterSymbol &) { return success(); }

  // Skip type-related declarations. These are absorbedby the types.
  LogicalResult visit(const slang::ast::TypeAliasType &) { return success(); }
  LogicalResult visit(const slang::ast::TypeParameterSymbol &) {
    return success();
  }
  LogicalResult visit(const slang::ast::TransparentMemberSymbol &) {
    return success();
  }

  // Handle instances.
  LogicalResult visit(const slang::ast::InstanceSymbol &instNode) {
    auto targetModule = context.convertModuleHeader(&instNode.body);
    if (!targetModule)
      return failure();

    builder.create<moore::InstanceOp>(
        loc, builder.getStringAttr(instNode.name),
        FlatSymbolRefAttr::get(targetModule.getSymNameAttr()));

    return success();
  }

  // Handle variables.
  LogicalResult visit(const slang::ast::VariableSymbol &varNode) {
    auto loweredType = context.convertType(*varNode.getDeclaredType());
    if (!loweredType)
      return failure();

    Value initial;
    if (varNode.getInitializer()) {
      initial = context.convertExpression(*varNode.getInitializer());
      if (!initial)
        return failure();
    }

    auto varOp = builder.create<moore::VariableOp>(
        loc, loweredType, builder.getStringAttr(varNode.name), initial);
    context.varSymbolTable.insert(varNode.name, varOp);
    return success();
  }

  // Handle nets.
  LogicalResult visit(const slang::ast::NetSymbol &netNode) {
    auto loweredType = context.convertType(*netNode.getDeclaredType());
    if (!loweredType)
      return failure();

    Value assignment;
    if (netNode.getInitializer()) {
      assignment = context.convertExpression(*netNode.getInitializer());
      if (!assignment)
        return failure();
    }

    auto netOp = builder.create<moore::NetOp>(
        loc, loweredType, builder.getStringAttr(netNode.name),
        builder.getStringAttr(netNode.netType.name), assignment);
    context.varSymbolTable.insert(netNode.name, netOp);
    return success();
  }

  // Handle ports.
  LogicalResult visit(const slang::ast::PortSymbol &portNode) {
    auto loweredType = context.convertType(portNode.getType());
    if (!loweredType)
      return failure();
    // TODO: Fix the `static_cast` here.
    builder.create<moore::PortOp>(
        loc, builder.getStringAttr(portNode.name),
        static_cast<moore::Direction>(portNode.direction));
    return success();
  }

  // Handle continuous assignments.
  LogicalResult visit(const slang::ast::ContinuousAssignSymbol &assignNode) {
    if (!context.convertExpression(assignNode.getAssignment()))
      return failure();
    return success();
  }

  // Handle statement blocks.
  LogicalResult visit(const slang::ast::StatementBlockSymbol &blockNode) {
    return context.convertStatementBlock(&blockNode);
  }

  // Handle procedures.
  LogicalResult visit(const slang::ast::ProceduralBlockSymbol &procNode) {
    auto procOp = builder.create<moore::ProcedureOp>(
        loc, convertProcedureKind(procNode.procedureKind));
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&procOp.getBodyBlock());
    return context.convertStatement(&procNode.getBody());
  }

  /// Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported construct: ")
        << slang::ast::toString(node.kind);
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Structure and Hierarchy Conversion
//===----------------------------------------------------------------------===//

/// Convert an entire Slang compilation to MLIR ops. This is the main entry
/// point for the conversion.
LogicalResult
Context::convertCompilation(slang::ast::Compilation &compilation) {
  const auto &root = compilation.getRoot();

  // Visit all top-level declarations in all compilation units. This does not
  // include instantiable constructs like modules, interfaces, and programs,
  // which are listed separately as top instances.
  for (auto *unit : root.compilationUnits) {
    for (const auto &member : unit->members()) {
      // Error out on all top-level declarations.
      auto loc = convertLocation(member.location);
      if (failed(member.visit(RootMemberVisitor(*this, loc))))
        return failure();
    }
  }

  // Prime the root definition worklist by adding all the top-level modules.
  SmallVector<const slang::ast::InstanceSymbol *> topInstances;
  for (auto *inst : root.topInstances)
    convertModuleHeader(&inst->body);

  // Convert all the root module definitions.
  while (!moduleWorklist.empty()) {
    auto *module = moduleWorklist.front();
    moduleWorklist.pop();
    if (failed(convertModuleBody(module)))
      return failure();
  }

  return success();
}

/// Convert a module and its ports to an empty module op in the IR. Also adds
/// the op to the worklist of module bodies to be lowered. This acts like a
/// module "declaration", allowing instances to already refer to a module even
/// before its body has been lowered.
moore::SVModuleOp
Context::convertModuleHeader(const slang::ast::InstanceBodySymbol *module) {
  if (auto op = moduleOps.lookup(module))
    return op;
  auto loc = convertLocation(module->location);
  OpBuilder::InsertionGuard g(builder);

  // We only support modules for now. Extension to interfaces and programs
  // should be trivial though, since they are essentially the same thing with
  // only minor differences in semantics.
  if (module->getDefinition().definitionKind !=
      slang::ast::DefinitionKind::Module) {
    mlir::emitError(loc, "unsupported construct: ")
        << module->getDefinition().getKindString();
    return {};
  }

  // Handle the port list.
  for (auto *symbol : module->getPortList()) {
    auto portLoc = convertLocation(symbol->location);
    auto *port = symbol->as_if<slang::ast::PortSymbol>();
    if (!port) {
      mlir::emitError(portLoc, "unsupported module port: `")
          << symbol->name << "` (" << slang::ast::toString(symbol->kind) << ")";
      return {};
    }
    LLVM_DEBUG(llvm::dbgs() << "- " << port->name << " "
                            << slang::ast::toString(port->direction) << "\n");
    if (auto *intSym = port->internalSymbol) {
      LLVM_DEBUG(llvm::dbgs() << "  - Internal symbol " << intSym->name << " ("
                              << slang::ast::toString(intSym->kind) << ")\n");
    }
    if (auto *expr = port->getInternalExpr()) {
      LLVM_DEBUG(llvm::dbgs() << "  - Internal expr "
                              << slang::ast::toString(expr->kind) << "\n");
    }
  }

  // Pick an insertion point for this module according to the source file
  // location.
  auto it = orderedRootOps.upper_bound(module->location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  // Create an empty module that corresponds to this module.
  auto moduleOp = builder.create<moore::SVModuleOp>(loc, module->name);
  orderedRootOps.insert(it, {module->location, moduleOp});
  moduleOp.getBodyRegion().emplaceBlock();

  // Add the module to the symbol table of the MLIR module, which uniquifies its
  // name as we'd expect.
  symbolTable.insert(moduleOp);

  // Schedule the body to be lowered.
  moduleWorklist.push(module);
  moduleOps.insert({module, moduleOp});
  return moduleOp;
}

/// Convert a module's body to the corresponding IR ops. The module op must have
/// already been created earlier through a `convertModuleHeader` call.
LogicalResult
Context::convertModuleBody(const slang::ast::InstanceBodySymbol *module) {
  LLVM_DEBUG(llvm::dbgs() << "Converting body of module " << module->name
                          << "\n");
  auto moduleOp = moduleOps.lookup(module);
  assert(moduleOp);
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // Create a new scope in a module. When the processing of a module is
  // terminated, the scope is destroyed and the mappings created in this scope
  // are dropped.
  SymbolTableScopeT varScope(varSymbolTable);

  for (auto &member : module->members()) {
    if (failed(walkMember(member)))
      return failure();
  }
  return success();
}

LogicalResult Context::walkMember(const slang::ast::Symbol &member) {
  LLVM_DEBUG(llvm::dbgs() << "- Handling " << slang::ast::toString(member.kind)
                          << "\n");
  auto loc = convertLocation(member.location);

  // Skip type-related declarations. These are absorbedby the types.
  if (member.kind == slang::ast::SymbolKind::TypeAlias ||
      member.kind == slang::ast::SymbolKind::TypeParameter ||
      member.kind == slang::ast::SymbolKind::TransparentMember)
    return success();

  // Skip semicolons.
  if (member.kind == slang::ast::SymbolKind::EmptyMember)
    return success();

  // Skip genvar.
  if (member.kind == slang::ast::SymbolKind::Genvar)
    return success();

  // Handle instances.
  if (auto *instAst = member.as_if<slang::ast::InstanceSymbol>()) {
    auto targetModule = convertModuleHeader(&instAst->body);
    if (!targetModule)
      return failure();
    builder.create<moore::InstanceOp>(
        loc, builder.getStringAttr(instAst->name),
        FlatSymbolRefAttr::get(SymbolTable::getSymbolName(targetModule)));
    return success();
  }

  // Handle variables.
  if (auto *varAst = member.as_if<slang::ast::VariableSymbol>()) {
    auto loweredType = convertType(*varAst->getDeclaredType());
    if (!loweredType)
      return failure();

    Value initial;
    if (varAst->getInitializer())
      initial = convertExpression(*varAst->getInitializer());

    auto varOp = builder.create<moore::VariableOp>(
        convertLocation(varAst->location), loweredType,
        builder.getStringAttr(varAst->name), initial);
    varSymbolTable.insert(varAst->name, varOp);
    createPostValue(loc);
    return success();
  }

  // Handle Nets.
  if (auto *netAst = member.as_if<slang::ast::NetSymbol>()) {
    auto loweredType = convertType(*netAst->getDeclaredType());
    if (!loweredType)
      return failure();

    Value assignment;
    if (netAst->getInitializer())
      assignment = convertExpression(*netAst->getInitializer());

    auto netOp = builder.create<moore::NetOp>(
        convertLocation(netAst->location), loweredType,
        builder.getStringAttr(netAst->name),
        builder.getStringAttr(netAst->netType.name), assignment);
    varSymbolTable.insert(netAst->name, netOp);
    return success();
  }

  // Handle Ports.
  if (auto *portAst = member.as_if<slang::ast::PortSymbol>()) {
    auto loweredType = convertType(portAst->getType());
    if (!loweredType)
      return failure();
    builder.create<moore::PortOp>(
        convertLocation(portAst->location),
        builder.getStringAttr(portAst->name),
        static_cast<moore::Direction>(portAst->direction));
    return success();
  }

  // Handle AssignOp.
  if (auto *assignAst = member.as_if<slang::ast::ContinuousAssignSymbol>()) {
    if (!convertExpression(assignAst->getAssignment()))
      return failure();
    return success();
  }

  // Handle StatementBlock.
  if (auto *stmtAst = member.as_if<slang::ast::StatementBlockSymbol>()) {
    if (failed(convertStatementBlock(stmtAst)))
      return failure();
    return success();
  }

  // Handle ProceduralBlock.
  if (auto *procAst = member.as_if<slang::ast::ProceduralBlockSymbol>()) {
    auto loc = convertLocation(procAst->location);
    auto procOp = builder.create<moore::ProcedureOp>(
        loc, convertProcedureKind(procAst->procedureKind));
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&procOp.getBodyBlock());
    if (failed(convertStatement(&procAst->getBody())))
      return failure();
    return success();
  }

  // Handle Parameters
  if (auto *paramAst = member.as_if<slang::ast::ParameterSymbol>()) {
    auto type = convertType(*paramAst->getDeclaredType());
    if (!type)
      return failure();
    auto value = *paramAst->getValue().integer().getRawPtr();
    auto namedConstantOp = builder.create<moore::NamedConstantOp>(
        convertLocation(paramAst->location), type,
        builder.getStringAttr(paramAst->name),
        paramAst->isLocalParam()
            ? moore::NamedConstAttr::get(getContext(),
                                         moore::NamedConst::LocalParameter)
            : moore::NamedConstAttr::get(getContext(),
                                         moore::NamedConst::Parameter),
        builder.getI64IntegerAttr(value));
    varSymbolTable.insert(paramAst->name, namedConstantOp);
    return success();
  }

  // Handle Specparam
  if (auto *spAst = member.as_if<slang::ast::SpecparamSymbol>()) {
    auto type = convertType(*spAst->getDeclaredType());
    if (!type)
      return failure();
    auto value = *spAst->getValue().integer().getRawPtr();
    auto namedConstantOp = builder.create<moore::NamedConstantOp>(
        convertLocation(spAst->location), type,
        builder.getStringAttr(spAst->name),
        moore::NamedConstAttr::get(getContext(),
                                   moore::NamedConst::SpecParameter),
        builder.getI64IntegerAttr(value));
    varSymbolTable.insert(spAst->name, namedConstantOp);
    return success();
  }

  // Handle GenerateBlock
  if (auto *genAst = member.as_if<slang::ast::GenerateBlockSymbol>()) {
    if (!genAst->isUninstantiated) {
      for (auto &mem : genAst->members()) {
        if (failed(walkMember(mem)))
          return failure();
      }
    }
    return success();
  }

  // Handle GenerateBlockArray
  if (auto *genArrAst = member.as_if<slang::ast::GenerateBlockArraySymbol>()) {
    for (const auto *genMember : genArrAst->entries) {
      if (failed(walkMember(genMember->asSymbol())))
        return failure();
    }
    return success();
  }

  // Otherwise just report that we don't support this SV construct yet and
  // skip over it. We'll want to make this an error, but in the early phases
  // we'll just want to cover ground as quickly as possible and skip over
  // things we don't support.
  mlir::emitWarning(loc, "unsupported construct ignored: ")
      << slang::ast::toString(member.kind);
  return success();
}

LogicalResult
Context::convertStatementBlock(const slang::ast::StatementBlockSymbol *stmt) {
  for (auto &member : stmt->members()) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Handling " << slang::ast::toString(member.kind) << "\n");
    auto loc = convertLocation(member.location);

    // Handle variables.
    if (auto *varAst = member.as_if<slang::ast::VariableSymbol>()) {
      auto loweredType = convertType(*varAst->getDeclaredType());
      if (!loweredType)
        return failure();

      Value initial;
      if (varAst->getInitializer())
        initial = convertExpression(*varAst->getInitializer());

      auto varOp = builder.create<moore::VariableOp>(
          convertLocation(varAst->location), loweredType,
          builder.getStringAttr(varAst->name), initial);
      varSymbolTable.insert(varAst->name, varOp);
      continue;
    }

    // Handle StatementBlock.
    if (auto *stmtAst = member.as_if<slang::ast::StatementBlockSymbol>()) {
      if (failed(convertStatementBlock(stmtAst)))
        return failure();
      continue;
    }

    mlir::emitWarning(loc, "unsupported construct ignored: ")
        << slang::ast::toString(member.kind);
  }
  return mlir::success();
}
