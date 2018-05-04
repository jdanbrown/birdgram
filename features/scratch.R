%%
suppressPackageStartupMessages(library(lookup))
library(repr)

trace('repr', where = repr)
trace('repr_text', where = repr)
trace('repr_text.default', where = repr)
trace('repr_html', where = repr)
trace('repr_html.default', where = repr)
trace('repr_function_generic', where = repr)
trace('repr_html.function', where = repr)
repr

%%
repr_html.function <- function(obj) { 'foo' }
rm(repr_html.function)
repr

%%
suppressPackageStartupMessages(library(lookup))
print(body)

%%
'lookup' %in% rownames(installed.packages())

%%
suppressPackageStartupMessages(library(lookup))
repr_html.function <- function(obj) {
  # Disable text/html so that jupyter/hydrogen/hydrogen-extras [which one?] falls back to text/plain
  NULL
}
repr_text.default <- function(obj) {
  # lookup overrides print.function, which print delegates to
  # - TODO Why must we override repr_text.default with the same definition that's already in repr?
	paste(capture.output(print(obj)), collapse = '\n')
}
# rm(repr_html.function)
# repr::repr
# repr_html.function(body)
# repr::repr_text
# repr::repr_text(body)
# capture.output(print(body))
body

%%
library(repr)
trace('repr_text.default', where = repr)
3
trace('repr.help_files_with_topic', where = repr)
trace('repr_text.help_files_with_topic', where = repr)
trace('repr_html.help_files_with_topic', where = repr)
repr.help_files_with_topic <- function(obj, ...) { 'baz' }
repr_html.help_files_with_topic <- function(obj, ...) { 'bar' }
repr_text.help_files_with_topic <- function(obj, ...) { 'foo' }
?trace
help(trace)
trace
sprintf('%s', help(trace))
repr::repr(?trace)
repr::repr_text(?trace)
repr::repr_html(?trace)

%%
repr_function_generic <- function(f, fmt, escape, high_wrap, norm_wrap, highlight) {
  'foo'
}
repr::repr

%%
trace('repr_html.function', quote(print(sys.calls())), where = repr)
repr

%%
trace(print.function)
trace(repr)
trace(repr_text)
trace(sprintf)

%%
print.function(repr)
print(repr)
repr

%%
repr_text(repr)
repr(repr)
repr
