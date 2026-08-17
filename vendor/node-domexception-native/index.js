// Drop-in replacement for the deprecated `node-domexception` package.
// The deprecation notice says: "Use your platform's native DOMException
// instead" — Node has shipped it as a global since v18, so this stub just
// re-exports that global. Consumers import it as a CommonJS default
// (`const DOMException = require("node-domexception")`), which this
// satisfies directly.
module.exports = globalThis.DOMException;
