import Foundation
import CoreGraphics

#if canImport(PDFium)
import PDFium
#endif

enum PDFiumError: LocalizedError {
    case frameworkUnavailable
    case invalidPath
    case loadFailed(code: Int32)
    case pageLoadFailed
    case renderFailed

    var errorDescription: String? {
        switch self {
        case .frameworkUnavailable:
            return "PDFium framework is not available. Add the PDFiumPackage dependency first."
        case .invalidPath:
            return "The selected file path is invalid."
        case .loadFailed(let code):
            return "PDFium failed to load the document (error code: \(code))."
        case .pageLoadFailed:
            return "PDFium failed to load the page."
        case .renderFailed:
            return "PDFium failed to render the page."
        }
    }
}

final class PDFiumService {
    static let shared = PDFiumService()

    private init() {
        #if canImport(PDFium)
        FPDF_InitLibrary()
        #endif
    }

    func pageCount(for url: URL) throws -> Int {
        guard let path = url.path.cString(using: .utf8) else {
            throw PDFiumError.invalidPath
        }

        #if canImport(PDFium)
        guard let document = FPDF_LoadDocument(path, nil) else {
            let code = Int32(FPDF_GetLastError())
            throw PDFiumError.loadFailed(code: code)
        }
        defer {
            FPDF_CloseDocument(document)
        }

        return Int(FPDF_GetPageCount(document))
        #else
        throw PDFiumError.frameworkUnavailable
        #endif
    }

    func renderFirstPage(for url: URL, maxDimension: CGFloat = 720) throws -> CGImage {
        guard let path = url.path.cString(using: .utf8) else {
            throw PDFiumError.invalidPath
        }

        #if canImport(PDFium)
        guard let document = FPDF_LoadDocument(path, nil) else {
            let code = Int32(FPDF_GetLastError())
            throw PDFiumError.loadFailed(code: code)
        }
        defer {
            FPDF_CloseDocument(document)
        }

        guard let page = FPDF_LoadPage(document, 0) else {
            throw PDFiumError.pageLoadFailed
        }
        defer {
            FPDF_ClosePage(page)
        }

        let pageWidth = CGFloat(FPDF_GetPageWidthF(page))
        let pageHeight = CGFloat(FPDF_GetPageHeightF(page))
        let maxPageDimension = max(pageWidth, pageHeight)
        let scale = maxPageDimension > 0 ? (maxDimension / maxPageDimension) : 1

        let targetWidth = max(1, Int(pageWidth * scale))
        let targetHeight = max(1, Int(pageHeight * scale))

        guard let bitmap = FPDFBitmap_Create(targetWidth, targetHeight, 1) else {
            throw PDFiumError.renderFailed
        }
        defer {
            FPDFBitmap_Destroy(bitmap)
        }

        FPDFBitmap_FillRect(bitmap, 0, 0, targetWidth, targetHeight, 0xFFFFFFFF)
        FPDF_RenderPageBitmap(bitmap, page, 0, 0, targetWidth, targetHeight, 0, 0)

        guard let buffer = FPDFBitmap_GetBuffer(bitmap) else {
            throw PDFiumError.renderFailed
        }

        let stride = Int(FPDFBitmap_GetStride(bitmap))
        let dataSize = stride * targetHeight
        let data = Data(bytes: buffer, count: dataSize)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.union(.premultipliedFirst)

        guard let provider = CGDataProvider(data: data as CFData) else {
            throw PDFiumError.renderFailed
        }

        guard let image = CGImage(
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: stride,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            throw PDFiumError.renderFailed
        }

        return image
        #else
        throw PDFiumError.frameworkUnavailable
        #endif
    }
}
