# DocuMind AI frontend

Next.js client for DocuMind AI. See the [root README](../README.md) for setup.

## Adding dependencies

CI runs `npm ci` on Linux, which requires the lock file to list every platform's optional
packages. An incremental `npm install <pkg>` on Windows or macOS can drop Linux-only
entries (such as `@emnapi/*`). After adding or upgrading a dependency, regenerate the lock
file from scratch:

```bash
rm -rf node_modules package-lock.json
npm install
```

To check it the way CI will: `npm ci --dry-run --os=linux --cpu=x64 --libc=glibc`.
