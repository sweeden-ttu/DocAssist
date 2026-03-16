//
//  ContentView.swift
//  DocAssist
//
//  Created by root on 3/16/26.
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var isImporterPresented = false
    @State private var selectedURL: URL?
    @State private var pageCount: Int?
    @State private var previewImage: CGImage?
    @State private var errorMessage: String?
    @State private var hasLoadedDefault = false

    private let defaultPDFPath = "/Users/sweeden/projects/DocAssist/templates/usda/FSA2001_250321V05LC (14).pdf"

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("DocAssist")
                .font(.system(size: 28, weight: .semibold))

            Text(pdfiumStatus)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Button("Select PDF") {
                isImporterPresented = true
            }

            if let selectedURL {
                Text("File: \(selectedURL.lastPathComponent)")
                    .font(.subheadline)
            }

            if let pageCount {
                Text("Pages: \(pageCount)")
                    .font(.title3)
            }

            if let previewImage {
                Text("Preview")
                    .font(.headline)

                Image(decorative: previewImage, scale: 1)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 520, maxHeight: 520)
                    .background(Color(.windowBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.secondary.opacity(0.2))
                    )
            }

            if let errorMessage {
                Text(errorMessage)
                    .foregroundStyle(.red)
            }

            Spacer()
        }
        .padding(24)
        .frame(minWidth: 520, minHeight: 360)
        .fileImporter(
            isPresented: $isImporterPresented,
            allowedContentTypes: [.pdf],
            allowsMultipleSelection: false
        ) { result in
            handleImport(result)
        }
        .onAppear {
            loadDefaultPDFIfNeeded()
        }
    }

    private var pdfiumStatus: String {
        #if canImport(PDFium)
        return "PDFium ready"
        #else
        return "PDFium module not linked"
        #endif
    }

    private func handleImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else {
                errorMessage = "No file was selected."
                return
            }
            loadPDF(from: url, requiresSecurityScope: true)
        case .failure(let error):
            errorMessage = error.localizedDescription
        }
    }

    private func loadDefaultPDFIfNeeded() {
        guard !hasLoadedDefault, selectedURL == nil else { return }
        hasLoadedDefault = true

        let url = URL(fileURLWithPath: defaultPDFPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            errorMessage = "Default PDF not found at: \(defaultPDFPath)"
            return
        }

        loadPDF(from: url, requiresSecurityScope: false)
    }

    private func loadPDF(from url: URL, requiresSecurityScope: Bool) {
        errorMessage = nil
        pageCount = nil
        previewImage = nil
        selectedURL = url

        let didStart = requiresSecurityScope ? url.startAccessingSecurityScopedResource() : false
        defer {
            if didStart {
                url.stopAccessingSecurityScopedResource()
            }
        }

        do {
            pageCount = try PDFiumService.shared.pageCount(for: url)
            previewImage = try PDFiumService.shared.renderFirstPage(for: url)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

#Preview {
    ContentView()
}
