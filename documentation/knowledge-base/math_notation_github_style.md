# Summary: Writing Mathematical Expressions on GitHub

When it comes to writing mathematical expressions on GitHub, the following points should be considered:

1. **Basic Syntax**:
   - Inline math: Use single dollar signs `$...$` to write inline mathematical expressions.
   - Display math: Use double dollar signs `$$...$$` to write expressions that are centered and on their own line.

2. **Escaping Dollar Signs**:
   - Use a backslash `\$` to escape dollar signs if you need to include them in text without triggering math mode.

3. **Common Operators and Functions**:
   - You can use common operators like `+`, `-`, `*`, `/`, `=`, and functions like `\sin`, `\cos`, `\tan`, `\log`, etc.

4. **Subscripts and Superscripts**:
   - Subscripts: Use the underscore `_` character. E.g., `x_i`.
   - Superscripts: Use the caret `^` character. E.g., `x^2`.

5. **Fractions**:
   - Use the `\frac{numerator}{denominator}` command to create fractions.

6. **Square Roots and nth Roots**:
   - Square roots: Use `\sqrt{x}`.
   - nth roots: Use `\sqrt[n]{x}`.

7. **Greek Letters**:
   - Use backslash followed by the name of the letter, like `\alpha`, `\beta`, `\gamma`, etc.

8. **Brackets and Parentheses**:
   - Use `\left` and `\right` to automatically size brackets and parentheses to the expression inside. E.g., `\left( \frac{a}{b} \right)`.

9. **Matrices**:
   - Use the `matrix` environment within `\begin{...}` and `\end{...}` to write matrices.

By following these guidelines, you can effectively write complex mathematical expressions in GitHub Markdown.
