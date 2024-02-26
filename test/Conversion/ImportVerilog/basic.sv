// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang


// CHECK-LABEL: moore.module @Variables
module Variables();
  // CHECK: %var1 = moore.variable : !moore.int
  // CHECK: %var2 = moore.variable %var1 : !moore.int
  int var1;
  int var2 = var1;
endmodule


// CHECK-LABEL: moore.module @Procedures
module Procedures();
  // CHECK: moore.procedure initial {
  initial;
  // CHECK: moore.procedure final {
  final begin end;
  // CHECK: moore.procedure always {
  always begin end;
  // CHECK: moore.procedure always_comb {
  always_comb begin end;
  // CHECK: moore.procedure always_latch {
  always_latch begin end;
  // CHECK: moore.procedure always_ff {
  always_ff @* begin end;
endmodule


// CHECK-LABEL: moore.module @Expressions {
module Expressions();
  // CHECK: %a = moore.variable : !moore.int
  // CHECK: %b = moore.variable : !moore.int
  // CHECK: %c = moore.variable : !moore.int
  int a, b, c;
  int unsigned u;
  bit [1:0][3:0] v;
  integer d, e, f;
  bit x;
  logic y;

  initial begin
    // CHECK: moore.constant 42 : !moore.int
    c = 42;

    // Unary operators

    // CHECK: moore.bpassign %c, %a : !moore.int
    c = a;
    // CHECK: moore.neg %a : !moore.int
    c = -a;
    // CHECK: [[TMP1:%.+]] = moore.conversion %v : !moore.packed<range<range<bit, 3:0>, 1:0>> -> !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP2:%.+]] = moore.neg [[TMP1]] : !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.packed<range<bit, 31:0>> -> !moore.int
    c = -v;
    // CHECK: moore.not %a : !moore.int
    c = ~a;

    // CHECK: moore.reduce_and %a : !moore.int -> !moore.bit
    x = &a;
    // CHECK: moore.reduce_and %d : !moore.integer -> !moore.logic
    y = &d;
    // CHECK: moore.reduce_or %a : !moore.int -> !moore.bit
    x = |a;
    // CHECK: moore.reduce_xor %a : !moore.int -> !moore.bit
    x = ^a;
    // CHECK: [[TMP:%.+]] = moore.reduce_and %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = ~&a;
    // CHECK: [[TMP:%.+]] = moore.reduce_or %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = ~|a;
    // CHECK: [[TMP:%.+]] = moore.reduce_xor %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = ~^a;
    // CHECK: [[TMP:%.+]] = moore.reduce_xor %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = ^~a;
    // CHECK: [[TMP:%.+]] = moore.bool_cast %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = !a;

    // CHECK: moore.bpassign %c, %a : !moore.int
    // CHECK: [[TMP1:%.+]] = moore.constant 1 : !moore.int
    // CHECK: [[TMP2:%.+]] = moore.add %a, [[TMP1]] : !moore.int
    // CHECK: moore.bpassign %a, [[TMP2]] : !moore.int
    c = a++;
    // CHECK: moore.bpassign %c, %a : !moore.int
    // CHECK: [[TMP1:%.+]] = moore.constant 1 : !moore.int
    // CHECK: [[TMP2:%.+]] = moore.sub %a, [[TMP1]] : !moore.int
    // CHECK: moore.bpassign %a, [[TMP2]] : !moore.int
    c = a--;
    // CHECK: [[TMP1:%.+]] = moore.constant 1 : !moore.int
    // CHECK: [[TMP2:%.+]] = moore.add %a, [[TMP1]] : !moore.int
    // CHECK: moore.bpassign %a, [[TMP2]] : !moore.int
    // CHECK: moore.bpassign %c, [[TMP2]] : !moore.int
    c = ++a;
    // CHECK: [[TMP1:%.+]] = moore.constant 1 : !moore.int
    // CHECK: [[TMP2:%.+]] = moore.sub %a, [[TMP1]] : !moore.int
    // CHECK: moore.bpassign %a, [[TMP2]] : !moore.int
    // CHECK: moore.bpassign %c, [[TMP2]] : !moore.int
    c = --a;

    // Binary operators

    // CHECK: moore.add %a, %b : !moore.int
    c = a + b;
    // CHECK: [[TMP1:%.+]] = moore.conversion %a : !moore.int -> !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP2:%.+]] = moore.conversion %v : !moore.packed<range<range<bit, 3:0>, 1:0>> -> !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP3:%.+]] = moore.add [[TMP1]], [[TMP2]] : !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.packed<range<bit, 31:0>> -> !moore.int
    c = a + v;
    // CHECK: moore.sub %a, %b : !moore.int
    c = a - b;
    // CHECK: moore.mul %a, %b : !moore.int
    c = a * b;
    // CHECK: moore.div %d, %e : !moore.integer
    f = d / e;
    // CHECK: moore.mod %d, %e : !moore.integer
    f = d % e;

    // CHECK: moore.and %a, %b : !moore.int
    c = a & b;
    // CHECK: moore.or %a, %b : !moore.int
    c = a | b;
    // CHECK: moore.xor %a, %b : !moore.int
    c = a ^ b;
    // CHECK: [[TMP:%.+]] = moore.xor %a, %b : !moore.int
    // CHECK: moore.not [[TMP]] : !moore.int
    c = a ~^ b;
    // CHECK: [[TMP:%.+]] = moore.xor %a, %b : !moore.int
    // CHECK: moore.not [[TMP]] : !moore.int
    c = a ^~ b;

    // CHECK: moore.eq %a, %b : !moore.int -> !moore.bit
    x = a == b;
    // CHECK: moore.eq %d, %e : !moore.integer -> !moore.logic
    y = d == e;
    // CHECK: moore.ne %a, %b : !moore.int -> !moore.bit
    x = a != b ;
    // CHECK: moore.case_eq %a, %b : !moore.int
    x = a === b;
    // CHECK: moore.case_ne %a, %b : !moore.int
    x = a !== b;
    // CHECK: moore.wildcard_eq %a, %b : !moore.int -> !moore.bit
    x = a ==? b;
    // CHECK: [[TMP:%.+]] = moore.conversion %a : !moore.int -> !moore.integer
    // CHECK: moore.wildcard_eq [[TMP]], %d : !moore.integer -> !moore.logic
    y = a ==? d;
    // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.int -> !moore.integer
    // CHECK: moore.wildcard_eq %d, [[TMP]] : !moore.integer -> !moore.logic
    y = d ==? b;
    // CHECK: moore.wildcard_eq %d, %e : !moore.integer -> !moore.logic
    y = d ==? e;
    // CHECK: moore.wildcard_ne %a, %b : !moore.int -> !moore.bit
    x = a !=? b;

    // CHECK: moore.ge %a, %b : !moore.int -> !moore.bit
    c = a >= b;
    // CHECK: moore.gt %a, %b : !moore.int -> !moore.bit
    c = a > b;
    // CHECK: moore.le %a, %b : !moore.int -> !moore.bit
    c = a <= b;
    // CHECK: moore.lt %a, %b : !moore.int -> !moore.bit
    c = a < b;

    // CHECK: [[A:%.+]] = moore.bool_cast %a : !moore.int -> !moore.bit
    // CHECK: [[B:%.+]] = moore.bool_cast %b : !moore.int -> !moore.bit
    // CHECK: moore.and [[A]], [[B]] : !moore.bit
    c = a && b;
    // CHECK: [[A:%.+]] = moore.bool_cast %a : !moore.int -> !moore.bit
    // CHECK: [[B:%.+]] = moore.bool_cast %b : !moore.int -> !moore.bit
    // CHECK: moore.or [[A]], [[B]] : !moore.bit
    c = a || b;
    // CHECK: [[A:%.+]] = moore.bool_cast %a : !moore.int -> !moore.bit
    // CHECK: [[B:%.+]] = moore.bool_cast %b : !moore.int -> !moore.bit
    // CHECK: [[NOT_A:%.+]] = moore.not [[A]] : !moore.bit
    // CHECK: moore.or [[NOT_A]], [[B]] : !moore.bit
    c = a -> b;
    // CHECK: [[A:%.+]] = moore.bool_cast %a : !moore.int -> !moore.bit
    // CHECK: [[B:%.+]] = moore.bool_cast %b : !moore.int -> !moore.bit
    // CHECK: [[NOT_A:%.+]] = moore.not [[A]] : !moore.bit
    // CHECK: [[NOT_B:%.+]] = moore.not [[B]] : !moore.bit
    // CHECK: [[BOTH:%.+]] = moore.and [[A]], [[B]] : !moore.bit
    // CHECK: [[NOT_BOTH:%.+]] = moore.and [[NOT_A]], [[NOT_B]] : !moore.bit
    // CHECK: moore.or [[BOTH]], [[NOT_BOTH]] : !moore.bit
    c = a <-> b;

    // CHECK: moore.shl %a, %b : !moore.int, !moore.int
    c = a << b;
    // CHECK: moore.shr %a, %b : !moore.int, !moore.int
    c = a >> b;
    // CHECK: moore.shl %a, %b : !moore.int, !moore.int
    c = a <<< b;
    // CHECK: moore.ashr %a, %b : !moore.int, !moore.int
    c = a >>> b;
    // CHECK: moore.shr %u, %b : !moore.int<unsigned>, !moore.int
    c = u >>> b;

    // CHECK: [[TMP1:%.+]] = moore.gt %a, %b : !moore.int -> !moore.bit
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.bit -> i1
    // CHECK: [[TMP3:%.+]] = scf.if [[TMP2]] -> (!moore.int) {
    // CHECK:   scf.yield %a : !moore.int
    // CHECK: } else {
    // CHECK:   scf.yield %b : !moore.int
    // CHECK: }
    // CHECK: moore.bpassign %c, [[TMP3]] : !moore.int
    c = a > b ? a : b;

    // CHECK: [[TMP1:%.+]] = moore.eq %a, %a : !moore.int -> !moore.bit
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.bit -> !moore.packed<range<logic, 31:0>>
    // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.packed<range<logic, 31:0>> -> !moore.int
    // CHECK: moore.bpassign %c, [[TMP3]] : !moore.int
    c = a inside {a};

    // CHECK: [[TMP1:%.+]] = moore.eq %a, %a : !moore.int -> !moore.bit
    // CHECK: [[TMP2:%.+]] = moore.eq %a, %b : !moore.int -> !moore.bit
    // CHECK: [[TMP3:%.+]] = moore.or [[TMP1]], [[TMP2]] : !moore.bit
    // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.bit -> !moore.packed<range<logic, 31:0>>
    // CHECK: [[TMP5:%.+]] = moore.conversion [[TMP4]] : !moore.packed<range<logic, 31:0>> -> !moore.int
    // CHECK: moore.bpassign %c, [[TMP5]] : !moore.int
    c = a inside {a, b};

    // CHECK: [[TMP1:%.+]] = moore.eq %a, %a : !moore.int -> !moore.bit
    // CHECK: [[TMP2:%.+]] = moore.eq %a, %b : !moore.int -> !moore.bit
    // CHECK: [[TMP3:%.+]] = moore.or [[TMP1]], [[TMP2]] : !moore.bit
    // CHECK: [[TMP4:%.+]] = moore.eq %a, %a : !moore.int -> !moore.bit
    // CHECK: [[TMP5:%.+]] = moore.or [[TMP3]], [[TMP4]] : !moore.bit
    // CHECK: [[TMP6:%.+]] = moore.eq %a, %b : !moore.int -> !moore.bit
    // CHECK: [[TMP7:%.+]] = moore.or [[TMP5]], [[TMP6]] : !moore.bit
    // CHECK: [[TMP8:%.+]] = moore.conversion [[TMP7]] : !moore.bit -> !moore.packed<range<logic, 31:0>>
    // CHECK: [[TMP9:%.+]] = moore.conversion [[TMP8]] : !moore.packed<range<logic, 31:0>> -> !moore.int
    // CHECK: moore.bpassign %c, [[TMP9]] : !moore.int
    c = a inside {a, b, a, b};
    
    // Assign operators

    // CHECK: moore.add %a, %b : !moore.int
    a += b;
    // CHECK: moore.sub %a, %b : !moore.int
    a -= b;
    // CHECK: moore.mul %a, %b : !moore.int
    a *= b;
    // CHECK: moore.div %f, %d : !moore.integer
    f /= d;
    // CHECK: moore.mod %f, %d : !moore.integer
    f %= d;
    // CHECK: moore.and %a, %b : !moore.int
    a &= b;
    // CHECK: moore.or %a, %b : !moore.int
    a |= b;
    // CHECK: moore.xor %a, %b : !moore.int
    a ^= b;
    // CHECK: moore.shl %a, %b : !moore.int, !moore.int
    a <<= b;
    // CHECK: moore.shl %a, %b : !moore.int, !moore.int
    a <<<= b;
    // CHECK: moore.shr %a, %b : !moore.int, !moore.int
    a >>= b;
    // CHECK: moore.ashr %a, %b : !moore.int, !moore.int
    a >>>= b;

  end
endmodule


// CHECK-LABEL: moore.module @Conversion {
module Conversion();
  // Implicit conversion.
  // CHECK: %a = moore.variable
  // CHECK: [[TMP:%.+]] = moore.conversion %a : !moore.shortint -> !moore.int
  // CHECK: %b = moore.variable [[TMP]]
  shortint a;
  int b = a;

  // Explicit conversion.
  // CHECK: [[TMP1:%.+]] = moore.conversion %a : !moore.shortint -> !moore.byte
  // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.byte -> !moore.int
  // CHECK: %c = moore.variable [[TMP2]]
  int c = byte'(a);

  // Sign conversion.
  // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.int -> !moore.packed<range<bit<signed>, 31:0>>
  // CHECK: %d1 = moore.variable [[TMP]]
  // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.int -> !moore.packed<range<bit, 31:0>>
  // CHECK: %d2 = moore.variable [[TMP]]
  bit signed [31:0] d1 = signed'(b);
  bit [31:0] d2 = unsigned'(b);

  // Width conversion.
  // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.int -> !moore.packed<range<bit<signed>, 18:0>>
  // CHECK: %e = moore.variable [[TMP]]
  bit signed [18:0] e = 19'(b);
endmodule


// CHECK-LABEL: moore.module @Assignments {
module Assignments();
  // CHECK: %a = moore.variable : !moore.int
  // CHECK: %b = moore.variable : !moore.int
  int a, b;

  initial begin
    // CHECK: moore.bpassign %a, %b : !moore.int
    a = b;
    // CHECK: moore.passign %a, %b : !moore.int
    a <= b;
    // CHECK: moore.pcassign %a, %b : !moore.int
    assign a = b;
  end
endmodule


// CHECK-LABEL: moore.module @Statements {
module Statements();
  // CHECK: %a = moore.variable : !moore.int
  // CHECK: %b = moore.variable : !moore.int
  int a, b, c;

  initial begin

    // CHECK: [[TMP:%.+]] = moore.bool_cast %a : !moore.int -> !moore.bit
    // CHECK: [[COND:%.+]] = moore.conversion [[TMP]] : !moore.bit -> i1
    // CHECK: scf.if [[COND]]
    if (a)
      ;

    // CHECK: [[TMP1:%.+]] = moore.eq %a, %a : !moore.int -> !moore.bit
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.bit -> i1
    // CHECK: scf.if [[TMP2]] {
    // CHECK: }
    // CHECK: [[TMP3:%.+]] = moore.eq %a, %b : !moore.int -> !moore.bit
    // CHECK: [[TMP4:%.+]] = moore.or [[TMP1]], [[TMP3]] : !moore.bit
    // CHECK: [[TMP5:%.+]] = moore.conversion [[TMP3]] : !moore.bit -> i1
    // CHECK: scf.if [[TMP5]] {
    // CHECK: }
    // CHECK: [[TMP6:%.+]] = moore.eq %a, %c : !moore.int -> !moore.bit
    // CHECK: [[TMP7:%.+]] = moore.or [[TMP4]], [[TMP6]] : !moore.bit
    // CHECK: [[TMP8:%.+]] = moore.conversion [[TMP6]] : !moore.bit -> i1
    // CHECK: scf.if [[TMP8]] {
    // CHECK: }
    // CHECK: [[TMP9:%.+]] = moore.not [[TMP7]] : !moore.bit
    // CHECK: [[TMP10:%.+]] = moore.conversion [[TMP9]] : !moore.bit -> i1
    // CHECK: scf.if [[TMP10]] {
    // CHECK: }
    case (a)
      a: ;
      b, c: ;
      default ;
    endcase

    // CHECK: [[TMP1:%.+]] = moore.eq %a, %a : !moore.int -> !moore.bit
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.bit -> i1
    // CHECK: scf.if [[TMP2]] {
    // CHECK: }
    // CHECK: [[TMP3:%.+]] = moore.eq %a, %b : !moore.int -> !moore.bit
    // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.bit -> i1
    // CHECK: scf.if [[TMP4]] {
    // CHECK: }
    // CHECK: [[TMP5:%.+]] = moore.eq %a, %c : !moore.int -> !moore.bit
    // CHECK: [[TMP6:%.+]] = moore.conversion [[TMP5]] : !moore.bit -> i1
    // CHECK: scf.if [[TMP6]] {
    // CHECK: }
    case (a)
      a: ;
      b, c: ;
    endcase
    
  end
endmodule


// CHECK-LABEL: moore.module @Generates {
module Generates();
  // CHECK: %a = moore.named_constant parameter 2 : !moore.packed<range<logic<signed>, 31:0>>
  // CHECK: %b = moore.named_constant localparam 3 : !moore.packed<range<logic<signed>, 31:0>>
  // CHECK：%sp = moore.named_constant specparam 4 : !moore.packed<range<logic<signed>, 31:0>>
  parameter a = 2;
  localparam b = 3;
  specparam sp = 4;
  genvar i;

  generate
    // CHECK: %i = moore.named_constant localparam 0 : !moore.integer
    // CHECK: [[TMP1:%.+]] = moore.conversion %i : !moore.integer -> !moore.int
    // CHECK: %c = moore.variable [[TMP1]] : !moore.int
    // CHECK: %i_0 = moore.named_constant name "i" localparam 1 : !moore.integer
    // CHECK: [[TMP2:%.+]] = moore.conversion %i_0 : !moore.integer -> !moore.int
    // CHECK: %c_1 = moore.variable name "c" [[TMP2]] : !moore.int
    for(i=0;i<a;i=i+1)begin
      int c = i;
    end

    // CHECK: [[TMP1:%.+]] = moore.constant 2 : !moore.int
    // CHECK: %d = moore.variable [[TMP1]] : !moore.int
    if(a == 2)begin
      int d = 2;
    end
    else begin
      int d = 3;
    end

    // CHECK: [[TMP1:%.+]] = moore.constant 2 : !moore.int
    // CHECK: %e = moore.variable [[TMP1]] : !moore.int
    case (a)
      2:begin
        int e = 2;
      end
      default:begin
        int e = 3;
      end
    endcase
  endgenerate
endmodule
