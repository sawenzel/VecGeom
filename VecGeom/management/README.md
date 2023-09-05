# Management utilities

## Logging

The `VECGEOM_LOG` macro creates a temporary object that sends the message to an
output handle, which by default prints with color to `std::clog` (which is
like `cerr` but with buffering).

The logger can be manually invoked using the `vecgeom::logger()` global
function. Its "call" operator takes the "Provenance" (file + line, or possibly
error code) and log level, and it returns the temporary object that will write
to output upon destruction.

The log levels should have particular meanings for consistency:

=========== =============================================================
Level       Description
=========== =============================================================
debug       Debugging messages
diagnostic  Diagnostics about current program execution
status      Program execution status (what stage is beginning)
info        Important informational messages
warning     Warnings about unusual events
error       Something went wrong, but execution can continue
critical    Something went terribly wrong: program should probably abort
=========== =============================================================

The `VECGEOM_LOG` environment variable, when set to one of these values,
determines the default output level, which by default is "status".

## Color output

The `vecgeom::color_code` function returns an ANSI string that can be used to
change the terminal's color and style. The `VECGEOM_COLOR` or `GTEST_COLOR`
environment variables can be used to explicitly enable or disable (by setting
to 1 or 0). The default ability to print color is determined by checking
whether the stderr stream is sent to the screen, and whether the `TERM`
environment variable is in the xterm family.

```c++
// Print a message in red
cerr << color_code('r') << "I'm angry!" << color_code(' ') << '\n'
     << "...and I'm normal\n";
```

## Environment variables

The `vecgeom::getenv` function provides thread-safe access to environment
variables. The map of variables that have been accessed at runtime is available
by interrogating `vecgeom::environment()`.
