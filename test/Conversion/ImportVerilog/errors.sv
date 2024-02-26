// RUN: circt-translate --import-verilog --verify-diagnostics --split-input-file %s
// REQUIRES: slang

// expected-error @below {{expected ';'}}
module Foo 4;
endmodule

// expected-note @below {{expanded from macro 'FOO'}}
`define FOO input
// expected-note @below {{expanded from macro 'BAR'}}
`define BAR `FOO
// expected-error @below {{expected identifier}}
module Bar(`BAR);
endmodule

// -----

module Foo;
  parameter A = 1;
  // expected-warning @below {{unsupported construct ignored}}
  defparam A = 233;
endmodule
