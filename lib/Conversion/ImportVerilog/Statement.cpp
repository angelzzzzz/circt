//===- Statement.cpp - Slang statement conversion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Symbol.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/ast/types/AllTypes.h"
#include "slang/ast/types/Type.h"
#include "slang/syntax/SyntaxVisitor.h"

using namespace circt;
using namespace ImportVerilog;

LogicalResult Context::visitConditionalStmt(
    const slang::ast::ConditionalStatement *conditionalStmt) {
  auto loc = convertLocation(conditionalStmt->sourceRange.start());

  Value cond = convertExpression(*conditionalStmt->conditions.begin()->expr);
  if (!cond)
    return failure();

  // The numeric value of the if expression is tested for being zero.
  // And if (expression) is equivalent to if (expression != 0).
  // So the following code is for handling `if (expression)`.
  if (auto condType = dyn_cast_or_null<moore::UnpackedType>(cond.getType())) {
    if (condType.isCastableToSimpleBitVector())
      cond = builder.create<moore::BoolCastOp>(loc, cond);
  }
  cond = builder.create<moore::ConversionOp>(loc, builder.getI1Type(), cond);

  auto ifOp = builder.create<mlir::scf::IfOp>(
      loc, cond, conditionalStmt->ifFalse != nullptr);
  OpBuilder::InsertionGuard guard(builder);

  builder.setInsertionPoint(ifOp.thenYield());
  if (failed(convertStatement(&conditionalStmt->ifTrue)))
    return failure();

  if (conditionalStmt->ifFalse) {
    builder.setInsertionPoint(ifOp.elseYield());
    if (failed(convertStatement(conditionalStmt->ifFalse)))
      return failure();
  }

  return success();
}

LogicalResult
Context::visitCaseStmt(const slang::ast::CaseStatement *caseStmt) {
  auto loc = convertLocation(caseStmt->sourceRange.start());
  auto caseExpr = convertExpression(caseStmt->expr);
  auto items = caseStmt->items;
  const auto *defaultCase = caseStmt->defaultCase;

  if (defaultCase != nullptr) {
    auto itemExpr = convertExpression(*items.front().expressions.front());
    Value preValue = builder.create<moore::EqOp>(loc, caseExpr, itemExpr);
    auto cond =
        builder.create<moore::ConversionOp>(loc, builder.getI1Type(), preValue);
    auto newIfOp = builder.create<mlir::scf::IfOp>(loc, cond);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(newIfOp.thenYield());
      if (failed(convertStatement(items.front().stmt)))
        return failure();
    }

    for (unsigned long i = 0; i < items.size(); i++) {
      auto itemStmt = items[i].stmt;
      for (unsigned long j = 0; j < items[i].expressions.size(); j++) {
        if (i == 0 && j == 0)
          continue;
        itemExpr = convertExpression(*items[i].expressions[j]);
        auto newEqOp = builder.create<moore::EqOp>(loc, caseExpr, itemExpr);
        preValue = builder.create<moore::LogicalOrOp>(loc, newEqOp.getType(),
                                                      preValue, newEqOp);
        auto cond = builder.create<moore::ConversionOp>(
            loc, builder.getI1Type(), newEqOp);
        auto newIfOp = builder.create<mlir::scf::IfOp>(loc, cond);
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(newIfOp.thenYield());
        if (failed(convertStatement(itemStmt)))
          return failure();
      }
    }

    auto notPreValue = builder.create<moore::NotOp>(loc, preValue);
    cond = builder.create<moore::ConversionOp>(loc, builder.getI1Type(),
                                               notPreValue);
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, cond);
    builder.setInsertionPoint(ifOp.thenYield());
    if (failed(convertStatement(defaultCase)))
      return failure();
  } else {
    for (auto item : items) {
      auto itemStmt = item.stmt;
      for (const auto *expr : item.expressions) {
        auto itemExpr = convertExpression(*expr);
        auto newEqOp = builder.create<moore::EqOp>(loc, caseExpr, itemExpr);
        auto cond = builder.create<moore::ConversionOp>(
            loc, builder.getI1Type(), newEqOp);
        auto ifOp = builder.create<mlir::scf::IfOp>(loc, cond);
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(ifOp.thenYield());
        if (failed(convertStatement(itemStmt)))
          return failure();
      }
    }
  }
  return success();
}

void Context::createPostValue(Location loc) {
  while (!postValueList.empty()) {
    auto value = postValueList.front();
    auto preValue = std::get<0>(value);
    auto one = builder.create<moore::ConstantOp>(loc, preValue.getType(), 1);
    auto postValue =
        std::get<1>(value)
            ? builder.create<moore::AddOp>(loc, preValue, one).getResult()
            : builder.create<moore::SubOp>(loc, preValue, one).getResult();
    builder.create<moore::BPAssignOp>(loc, preValue, postValue);
    postValueList.pop();
  }
}

// It can handle the statements like case, conditional(if), for loop, and etc.
LogicalResult
Context::convertStatement(const slang::ast::Statement *statement) {
  auto loc = convertLocation(statement->sourceRange.start());
  switch (statement->kind) {
  case slang::ast::StatementKind::Empty:
    return success();
  case slang::ast::StatementKind::List:
    for (auto *stmt : statement->as<slang::ast::StatementList>().list)
      if (failed(convertStatement(stmt)))
        return failure();
    break;
  case slang::ast::StatementKind::Block:
    return convertStatement(&statement->as<slang::ast::BlockStatement>().body);
  case slang::ast::StatementKind::ExpressionStatement: {
    auto value = convertExpression(
        statement->as<slang::ast::ExpressionStatement>().expr);
    createPostValue(loc);
    return success(value);
  }
  case slang::ast::StatementKind::VariableDeclaration:
    return mlir::emitError(loc, "unsupported statement: variable declaration");
  case slang::ast::StatementKind::Return:
    return mlir::emitError(loc, "unsupported statement: return");
  case slang::ast::StatementKind::Break:
    return mlir::emitError(loc, "unsupported statement: break");
  case slang::ast::StatementKind::Continue:
    return mlir::emitError(loc, "unsupported statement: continue");
  case slang::ast::StatementKind::Case: {
    return visitCaseStmt(&statement->as<slang::ast::CaseStatement>());
  }
  case slang::ast::StatementKind::PatternCase:
    return mlir::emitError(loc, "unsupported statement: pattern case");
  case slang::ast::StatementKind::ForLoop:
    return mlir::emitError(loc, "unsupported statement: for loop");
  case slang::ast::StatementKind::RepeatLoop:
    return mlir::emitError(loc, "unsupported statement: repeat loop");
  case slang::ast::StatementKind::ForeachLoop:
    return mlir::emitError(loc, "unsupported statement: foreach loop");
  case slang::ast::StatementKind::WhileLoop:
    return mlir::emitError(loc, "unsupported statement: while loop");
  case slang::ast::StatementKind::DoWhileLoop:
    return mlir::emitError(loc, "unsupported statement: do while loop");
  case slang::ast::StatementKind::ForeverLoop:
    return mlir::emitError(loc, "unsupported statement: forever loop");
  case slang::ast::StatementKind::Timed:
    if (failed(visitTimingControl(
            &statement->as<slang::ast::TimedStatement>().timing)))
      return failure();
    if (failed(convertStatement(
            &statement->as<slang::ast::TimedStatement>().stmt)))
      return failure();
    break;
  case slang::ast::StatementKind::ImmediateAssertion:
    return mlir::emitError(loc, "unsupported statement: immediate assertion");
  case slang::ast::StatementKind::ConcurrentAssertion:
    return mlir::emitError(loc, "unsupported statement: concurrent assertion");
  case slang::ast::StatementKind::DisableFork:
    return mlir::emitError(loc, "unsupported statement: disable fork");
  case slang::ast::StatementKind::Wait:
    return mlir::emitError(loc, "unsupported statement: wait");
  case slang::ast::StatementKind::WaitFork:
    return mlir::emitError(loc, "unsupported statement: wait fork");
  case slang::ast::StatementKind::WaitOrder:
    return mlir::emitError(loc, "unsupported statement: wait order");
  case slang::ast::StatementKind::EventTrigger:
    return mlir::emitError(loc, "unsupported statement: event trigger");
  case slang::ast::StatementKind::ProceduralAssign:
    return success(convertExpression(
        statement->as<slang::ast::ProceduralAssignStatement>().assignment));
  case slang::ast::StatementKind::ProceduralDeassign:
    return mlir::emitError(loc, "unsupported statement: procedural deassign");
  case slang::ast::StatementKind::RandCase:
    return mlir::emitError(loc, "unsupported statement: rand case");
  case slang::ast::StatementKind::RandSequence:
    return mlir::emitError(loc, "unsupported statement: rand sequence");
  case slang::ast::StatementKind::Conditional:
    return visitConditionalStmt(
        &statement->as<slang::ast::ConditionalStatement>());
  default:
    mlir::emitRemark(loc, "unsupported statement: ")
        << slang::ast::toString(statement->kind);
    return failure();
  }

  return success();
}
