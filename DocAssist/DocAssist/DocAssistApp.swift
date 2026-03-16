//
//  DocAssistApp.swift
//  DocAssist
//
//  Created by root on 3/16/26.
//

import SwiftUI
import SwiftData
import UniformTypeIdentifiers

@main
struct DocAssistApp: App {
    var body: some Scene {
        DocumentGroup(editing: .itemDocument, migrationPlan: DocAssistMigrationPlan.self) {
            ContentView()
        }
    }
}

extension UTType {
    static var itemDocument: UTType {
        UTType(importedAs: "com.example.item-document")
    }
}

struct DocAssistMigrationPlan: SchemaMigrationPlan {
    static var schemas: [VersionedSchema.Type] = [
        DocAssistVersionedSchema.self,
    ]

    static var stages: [MigrationStage] = [
        // Stages of migration between VersionedSchema, if required.
    ]
}

struct DocAssistVersionedSchema: VersionedSchema {
    static var versionIdentifier = Schema.Version(1, 0, 0)

    static var models: [any PersistentModel.Type] = [
        Item.self,
    ]
}
