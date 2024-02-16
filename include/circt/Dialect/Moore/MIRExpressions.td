//===- MIRExpressions.td - Moore MIR expression ops --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for Moore MIR expressions.
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/Moore/MooreTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// Base class for binary operators.
class BinOp<string mnemonic, list<Trait> traits = []> :
      MIROp<mnemonic, traits> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, UnitAttr:$twoState);
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$lhs `,` $rhs (`bin` $twoState^)? attr-dict `:` functional-type($args, $results)";
}

//===----------------------------------------------------------------------===//
// Shift operations
//===----------------------------------------------------------------------===//

class ShiftOp<string name> : MIROp<name, [
    Pure, 
    TypesMatchWith<"value and result types must match",
                   "value", "result", "$_self">
]> {
  let arguments = (ins SimpleBitVectorType:$value,
                       SimpleBitVectorType:$amount,
                       UnitAttr:$arithmetic);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    ( `arithmetic` $arithmetic^ )? $value `,` $amount attr-dict
    `:` type($value) `,` type($amount)
  }];
}

def ShlOp : ShiftOp<"shl"> {
  let summary = "A logical or arithmetic left-shift expression";
  let description = [{
    This operation represents the SystemVerilog logical and arithmetic
    left-shift expressions `<<` and `<<<`.
    See IEEE 1800-2017 §11.4.10 "Shift operators".

    The value to be shifted and the amount must be simple bit vector types.
    The shift result is of the same type as the input value.

    The logical and arithmetic shift both insert zeros in place of the shifted
    bits. 
  }];
}

def ShrOp : ShiftOp<"shr"> {
  let summary = "A logical or arithmetic right-shift expression";
  let description = [{
    This operation represents the SystemVerilog logical and arithmetic
    right-shift expressions `>>` and `>>>`.
    See IEEE 1800-2017 §11.4.10 "Shift operators".

    The value to be shifted and the amount must be simple bit vector types.
    The shift result is of the same type as the input value.

    The logical shift always inserts zeros in place of the shifted bits.
    The arithmetic shift inserts zeros if the result type is unsigned or the 
    MSB (sign bit) if the result type is signed.
  }];
}

//===---------------------------------------------------------------------===//
// Equality operations
//===---------------------------------------------------------------------===//

class EqualOp<string name> : MIROp<name, [
  Pure,
  Commutative
]> {
  let arguments = (ins SimpleBitVectorType:$lhs,
                       SimpleBitVectorType:$rhs,
                       UnitAttr:$mode);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    ( `case` $mode^ )? $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
  }];
}

def EqualityOp : EqualOp<"eq">{
  let summary = "A logical or case equality expression";
  let description = [{
    This operation represents the SystemVerilog logical and case
    equality expressions `==` and `===`.
    See IEEE1800-2017 11.4.5 "Equality operators".

    a == ba equal to b, result can be unknown.
    a === ba equal to b, including x and z.

    The operators compare operands bit for bit. As with the relational
    operators, the result shall be 0 if comparison fails and 1 if it succeeds.
  }];
}

def InEqualityOp : EqualOp<"ne">{
  let summary = "A logical or case inequality expression";
  let description = [{
    This operation represents the SystemVerilog logical and case
    inequality expressions `!=` and `!==`.
    See IEEE1800-2017 11.4.5 "Equality operators".

    a != ba not equal to b, result can be unknown.
    a !== ba not equal to b, including x and z.
  }];
}

//===---------------------------------------------------------------------===//
// Logical operations
//===---------------------------------------------------------------------===//

def LogicalAnd : I32EnumAttrCase<"LogicalAnd", 0, "and">;
def LogicalOr : I32EnumAttrCase<"LogicalOr", 1, "or">;
def LogicalImplication : I32EnumAttrCase<"LogicalImplication", 2, "impl">;
def LogicalEquivalence : I32EnumAttrCase<"LogicalEquivalence", 3, "equiv">;

def LogicalAttr : I32EnumAttr<"Logic", "Logical predicate", 
       [LogicalAnd, LogicalOr, LogicalImplication, LogicalEquivalence]>{
    let cppNamespace = "circt::moore";
}

def LogicalOp : MIROp<"logic", [
  Pure,
  TypesMatchWith<"lhs and result types must match",
                   "lhs", "result", "$_self">
]> {
  let summary = "Definitions of logical operators";
  let description = [{
    The result of the evaluation of a logical operation shall be `1`, `0`, 
    or, if the result is ambiguous, the unkonw value `x`.
    See IEEE Std 1800-2017 11.4.7.
  }];
  let arguments = (ins LogicalAttr:$logic,
                       SimpleBitVectorType:$lhs,
                       SimpleBitVectorType:$rhs);
  let results = (outs SimpleBitVectorType:$result);

  let assemblyFormat = [{
    $logic $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
  }];
}
