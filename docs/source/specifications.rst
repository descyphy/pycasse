Writing Specifications for PyCASSE
================================

PyCASSE currently supports specifications written in Signal Temporal Logic (STL) and Stochastic Temporal Logic (StSTL).

PyCASSE Syntax
--------------------------
.. Syntax for PyCASSE

An STL [Maler04]_ or an StSTL [Nuzzo19]_ formula can be written in PyCASSE using the following PyCASSE syntax in `Backus-Naur form <https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form>`_::

   <boolean> ::= "TRUE" | "FALSE"

   <digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

   <letter> ::= "A" | "B" | "C" | "D" | "E" | "H" | "I" | "J" | "K" | "L" | "M"
              | "N" | "O" | "Q" | "R" | "S" | "T" | "V" | "W" | "X" | "Y" | "Z"
              | "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m"
              | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"

   <integer> ::= <digit> | <digit> <integer>

   <interval> ::= "[" <integer> "," <integer> "]"

   <variable> ::= <letter> | <letter> <variable> | <variable> <integer>

   <expression> ::= <integer> | <variable> | <integer> "*" <variable>
                  | <expression> "+" <expression>
                  | <expression> "-" <expression>

   <inequality> ::= <expression> "<" <expression>
                  | <expression> ">" <expression>
                  | <expression> "<=" <expression>
                  | <expression> "=>" <expression>
                  | <expression> "==" <expression>

   <predicate> ::= <boolean> | <inequality> | "P(" <inequality> ") => 0." <integer>
                 
   <formula> ::= "(" <predicate> ")"
               | "(!" <formula> ")"
               | "(" <formula> " & " <formula> ")"
               | "(" <formula> " | " <formula> ")"
               | "(" <formula> " -> " <formula> ")"

               | "(G" <interval> " " <formula> ")"
               | "(F" <interval> " " <formula> ")"
               | "(" <formula> " U" <interval> " " <formula> ")"

For example, 'Globally from time step :math:`0` to :math:`3`, :math:`2x \geq 3` implies eventually from time step :math:`4` to :math:`5`, the probability of :math:`5 \leq y` larger than or equal to :math:`0.95`' can be written as an StSTL formula: :math:`\mathbf{G}_{[0,3]}(x \geq 3) \rightarrow \mathbf{F}_{[4,5]}(P\{ 5 \leq y \} \geq 0.95)` and can be written as a formula in PySTL: 

.. code-block:: python

   ((G[0,3] (2*x => 3)) -> (F[4,5] (P(5 <= y) => 0.95)))