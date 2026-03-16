@testable import DocAssist
import Testing

struct PDFiumModuleTests {
    @Test
    func pdfiumModuleIsAvailable() async throws {
        #if canImport(PDFium)
        _ = PDFiumService.shared
        #else
        Issue.record("PDFium module is not available. Ensure PDFiumPackage is linked.")
        #endif
    }
}
