# DocAssist PDFium Integration Plan

## Project Overview

- **macOS Client**: SwiftUI app using PDFium for PDF rendering/editing
- **Rust Server**: Backend service using pdfium-render for PDF processing
- **Communication**: gRPC between client and server

---

## Architecture

```
┌─────────────────────┐     gRPC      ┌─────────────────────┐
│   macOS Client     │ ◄─────────────► │   Rust Server       │
│   (Swift/SwiftUI)  │                │   (pdfium-render)   │
└─────────────────────┘                └─────────────────────┘
         │                                      │
         │ PDFium XCFramework                  │ PDFium dynamic lib
         ▼                                      ▼
┌─────────────────────┐                ┌─────────────────────┐
│  PDFKit (built-in)  │                │  libpdfium.dylib   │
│  or PDFium XCFrame  │                │  (runtime loading)  │
└─────────────────────┘                └─────────────────────┘
```

---

## Current Status

### Completed
- ✅ Rust server project initialized at `/Users/sweeden/projects/DocAssist/`
- ✅ gRPC service defined in `proto/pdf_service.proto`
- ✅ Server implementation with pdfium-render in `src/main.rs`
- ✅ Rust server builds successfully (requires PDFium dylib at runtime)

### Pending
- ⏳ Build PDFium from source (complex, requires full Chromium build setup)
- ⏳ Add XCFramework to macOS client
- ⏳ Implement gRPC client in macOS client

---

## Alternative Approach: Use Pre-built Binaries

Given the complexity of building PDFium from source, the recommended approach is:

### For Rust Server
The server uses pdfium-render which can load PDFium dynamically at runtime. Place `libpdfium.dylib` in the same directory as the executable or use system library.

### For macOS Client
Use pre-built XCFramework from: https://github.com/espresso3389/pdfium-xcframework

---

## Phase 1: Get PDFium Dynamic Library

### Option A: Download Pre-built (Recommended)
```bash
# Download macOS arm64 build
curl -L -o libpdfium.dylib \
  "https://github.com/bblanchon/pdfium-binaries/releases/download/latest/pdfium-mac-arm64.tgz"
tar -xzf pdfium-mac-arm64.tgz
# Copy libpdfium.dylib to same directory as your server executable
```

### Option B: Build from Source (Complex)
Requires full Chromium build infrastructure. See Phase 1 below.

---

## Phase 2: Run Rust Server

```bash
cd /Users/sweeden/projects/DocAssist

# Ensure libpdfium.dylib is in the current directory or set the path
export PDFIUM_DYNAMIC_LIB_PATH=/path/to/pdfium/dylib

# Run the server
cargo run --release

# Or build
cargo build --release
```

The server listens on `[::1]:50051` (localhost only for security).

---

## Phase 3: Set Up macOS Client

### Option A: Use PDFKit (Simplest - Built into macOS)
For basic PDF viewing, use Apple's PDFKit which is built into macOS:
```swift
import PDFKit
// Use PDFView for PDF display
```

### Option B: Use Pre-built XCFramework
1. Download from: https://github.com/espresso3389/pdfium-xcframework
2. Add to Xcode project
3. Configure bridging header

### Option C: Use Swift Package Manager
Add to your `Package.swift` or use Xcode's SPM integration:
```
https://github.com/espresso3389/pdfium-xcframework
```

---

## Phase 4: gRPC Client

### For Swift/macOS Client
Use `grpc-swift` or generate Swift from proto:

```swift
// Using grpc-swift
import GRPC
import NIO

let channel = ClientConnection
    .insecure(group: PlatformEventLoopGroup(loopCount: 1)!)
    .withConnectionTimeout(minimum: .seconds(1))
    .connect(host: "localhost", port: 50051)

let client = PdfServiceClient(channel: channel)

// Make requests
let request = RenderPageRequest.with {
    $0.pdfData = pdfData
    $0.pageIndex = 0
    $0.width = 612
    $0.height = 792
}

let response = try client.renderPage(request)
```

---

## File Structure

```
/Users/sweeden/projects/DocAssist/
├── Cargo.toml              # Rust project config
├── build.rs                # gRPC build script
├── proto/
│   └── pdf_service.proto   # gRPC service definition
├── src/
│   └── main.rs            # Server implementation
├── PLAN.md                # This plan
└── DocAssist/             # macOS client (Xcode project)
    └── ...
```

---

## Key Dependencies

### Rust Server
- `pdfium-render`: PDF processing
- `tonic`: gRPC server
- `prost`: Protocol buffers
- `tokio`: Async runtime

### macOS Client
- `grpc-swift` or `swift-protobuf`: For gRPC client
- Optional: PDFium XCFramework for advanced features

---

## Notes

1. **Security**: Server binds to localhost only. For production, add TLS and authentication.
2. **PDFium License**: BSD-style, compatible with App Store distribution.
3. **App Store**: Enable Hardened Runtime and code signing.
