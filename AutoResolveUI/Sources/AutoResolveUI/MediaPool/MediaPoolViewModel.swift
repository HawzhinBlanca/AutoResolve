import AppKit
// AUTORESOLVE V3.0 - MEDIA POOL VIEW MODEL
// Enterprise-grade media pool management with real-time sync and collaboration

import Foundation
import SwiftUI
import AVFoundation
import Combine
import CoreData
import UniformTypeIdentifiers

// MARK: - Media Pool View Model
@MainActor
final class MediaPoolViewModel: ObservableObject {
    // MARK: - Published Properties
    @Published var items: [MediaPoolItem] = []
    @Published var selectedItems: Set<UUID> = []
    @Published var filteredItems: [MediaPoolItem] = []
    @Published var isLoading = false
    @Published var loadingProgress: Double = 0
    @Published var searchText = ""
    @Published var sortOrder: SortOrder = .dateModified
    @Published var filterType: FilterType = .all
    @Published var viewMode: ViewMode = .thumbnail
    @Published var groupingMode: GroupingMode = .none
    @Published var error: MediaPoolError?
    
    // Collaboration
    @Published var activeUsers: [CollaboratingUser] = []
    @Published var lockedItems: Set<UUID> = []
    @Published var recentChanges: [MediaChange] = []
    
    // Performance metrics
    @Published var thumbnailCacheSize: Int64 = 0
    @Published var itemCount: Int = 0
    @Published var selectedCount: Int = 0
    
    // MARK: - Types
    enum SortOrder: String, CaseIterable {
        case name = "Name"
        case dateAdded = "Date Added"
        case dateModified = "Date Modified"
        case size = "Size"
        case duration = "Duration"
        case type = "Type"
        
        @MainActor func sortValue(for item: MediaPoolItem) -> String {
            switch self {
            case .name: return item.name
            case .dateAdded: return "\(item.dateCreated.timeIntervalSince1970)"
            case .dateModified: return "\(item.dateModified.timeIntervalSince1970)"
            case .size: return "\(item.fileSize)"
            case .duration: return "\(item.duration ?? 0)"
            case .type: return item.url.pathExtension
            }
        }
    }
    
    enum FilterType: String, CaseIterable {
        case all = "All Media"
        case video = "Video Only"
        case audio = "Audio Only"
        case images = "Images"
        case recent = "Recent (24h)"
        case unused = "Unused"
        case inTimeline = "In Timeline"
    }
    
    enum ViewMode: String, CaseIterable {
        case list = "List"
        case thumbnail = "Thumbnail"
        case filmstrip = "Filmstrip"
        case detail = "Detail"
    }
    
    enum GroupingMode: String, CaseIterable {
        case none = "No Grouping"
        case type = "By Type"
        case date = "By Date"
        case scene = "By Scene"
        case folder = "By Folder"
    }
    
    struct CollaboratingUser: Identifiable {
        public let id = UUID()
        let username: String
        let avatar: NSImage?
        let color: Color
        let currentSelection: UUID?
        let lastActivity: Date
        var isActive: Bool
    }
    
    struct MediaChange: Identifiable {
        public let id = UUID()
        let itemId: UUID
        let userId: String
        let changeType: ChangeType
        let timestamp: Date
        let description: String
        
        enum ChangeType {
            case added, removed, modified, locked, unlocked
        }
    }
    
    enum MediaPoolError: LocalizedError {
        case importFailed(String)
        case deleteFailed(String)
        case syncFailed(String)
        case databaseError(String)
        case accessDenied(String)
        
        var errorDescription: String? {
            switch self {
            case .importFailed(let msg): return "Import failed: \(msg)"
            case .deleteFailed(let msg): return "Delete failed: \(msg)"
            case .syncFailed(let msg): return "Sync failed: \(msg)"
            case .databaseError(let msg): return "Database error: \(msg)"
            case .accessDenied(let msg): return "Access denied: \(msg)"
            }
        }
    }
    
    // MARK: - Private Properties
    private let mediaImportManager = MediaImportManager()
    private let thumbnailCache = ThumbnailCacheManager.shared
    private let backendService: BackendService
    private var cancellables = Set<AnyCancellable>()
    private let undoManager = UndoManager()
    private let fileWatcher = FileSystemWatcher()
    
    // Database
    private var persistentContainer: NSPersistentContainer?
    private let databaseQueue = DispatchQueue(label: "com.autoresolve.mediapool.db", qos: .userInitiated)
    
    // Real-time sync
    private var syncWebSocket: URLSessionWebSocketTask?
    private let syncQueue = DispatchQueue(label: "com.autoresolve.mediapool.sync", qos: .userInitiated)
    
    // Performance
    private let itemsCache = NSCache<NSString, MediaPoolItem>()
    private let searchDebouncer = Debouncer(delay: 0.3)
    
    // MARK: - Initialization
    init(backendService: BackendService = BackendService.shared) {
        self.backendService = backendService
        setupBindings()
        setupDatabase()
        setupRealTimeSync()
        loadMediaItems()
    }
    
    // MARK: - Setup
    private func setupBindings() {
        // Search filter
        $searchText
            .removeDuplicates()
            .sink { [weak self] searchText in
                self?.searchDebouncer.debounce {
                    Task { @MainActor in
                        self?.applyFilters()
                    }
                }
            }
            .store(in: &cancellables)
        
        // Sort order change
        $sortOrder
            .removeDuplicates()
            .sink { [weak self] _ in
                self?.applySorting()
            }
            .store(in: &cancellables)
        
        // Filter type change
        $filterType
            .removeDuplicates()
            .sink { [weak self] _ in
                self?.applyFilters()
            }
            .store(in: &cancellables)
        
        // Selection change
        $selectedItems
            .sink { [weak self] selection in
                self?.selectedCount = selection.count
                self?.broadcastSelectionChange(selection)
            }
            .store(in: &cancellables)
        
        // Import manager
        mediaImportManager.$isImporting
            .assign(to: &$isLoading)
        
        mediaImportManager.$importProgress
            .assign(to: &$loadingProgress)
        
        // File system watcher
        fileWatcher.onChange = { [weak self] changedFiles in
            Task { @MainActor in
                self?.handleFileSystemChanges(changedFiles)
            }
        }
    }
    
    private func setupDatabase() {
        // Setup Core Data for persistence
        persistentContainer = NSPersistentContainer(name: "MediaPool")
        
        let description = NSPersistentStoreDescription()
        description.type = NSSQLiteStoreType
        description.url = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first?
            .appendingPathComponent("AutoResolve")
            .appendingPathComponent("MediaPool.sqlite")
        
        persistentContainer?.persistentStoreDescriptions = [description]
        
        persistentContainer?.loadPersistentStores { [weak self] _, error in
            if let error = error {
                self?.error = .databaseError(error.localizedDescription)
            }
        }
    }
    
    private func setupRealTimeSync() {
        // Setup WebSocket for real-time collaboration
        guard let url = URL(string: "ws://localhost:8000/ws/mediapool") else { return }
        
        let session = URLSession(configuration: .default)
        syncWebSocket = session.webSocketTask(with: url)
        syncWebSocket?.resume()
        
        receiveWebSocketMessage()
    }
    
    // MARK: - Public Methods
    
    // Import Methods
    func importFiles(_ urls: [URL]) {
        Task {
            isLoading = true
            defer { isLoading = false }
            
            // Validate URLs
            let validUrls = urls.filter { url in
                FileManager.default.fileExists(atPath: url.path)
            }
            
            // Import files
            await mediaImportManager.importFiles(validUrls)
            
            // Add imported items
            for url in validUrls {
                if let item = try? await createMediaItem(from: url) {
                    addItem(item)
                }
            }
            
            // Sync with backend
            await syncWithBackend()
        }
    }
    
    func addItem(_ item: MediaPoolItem) {
        guard !items.contains(where: { $0.id == item.id }) else { return }
        
        items.append(item)
        itemCount = items.count
        
        // Cache item
        itemsCache.setObject(item, forKey: item.id.uuidString as NSString)
        
        // Add to database
        saveToDatabase(item)
        
        // Broadcast change
        broadcastChange(.added, for: item)
        
        // Apply filters
        applyFilters()
        
        // Register undo
        undoManager.registerUndo(withTarget: self) { target in
            Task { @MainActor in
                target.removeItem(item)
            }
        }
        undoManager.setActionName("Add \(item.name)")
    }
    
    func removeItem(_ item: MediaPoolItem) {
        items.removeAll { $0.id == item.id }
        selectedItems.remove(item.id)
        itemCount = items.count
        
        // Remove from cache
        itemsCache.removeObject(forKey: item.id.uuidString as NSString)
        
        // Remove from database
        deleteFromDatabase(item)
        
        // Broadcast change
        broadcastChange(.removed, for: item)
        
        // Apply filters
        applyFilters()
        
        // Register undo
        undoManager.registerUndo(withTarget: self) { target in
            Task { @MainActor in
                target.addItem(item)
            }
        }
        undoManager.setActionName("Remove \(item.name)")
    }
    
    func removeSelected() {
        let itemsToRemove = items.filter { selectedItems.contains($0.id) }
        
        for item in itemsToRemove {
            removeItem(item)
        }
        
        selectedItems.removeAll()
    }
    
    // Selection Methods
    func selectAll() {
        selectedItems = Set(filteredItems.map { $0.id })
    }
    
    func deselectAll() {
        selectedItems.removeAll()
    }
    
    func toggleSelection(for item: MediaPoolItem) {
        if selectedItems.contains(item.id) {
            selectedItems.remove(item.id)
        } else {
            selectedItems.insert(item.id)
        }
    }
    
    func selectRange(from: MediaPoolItem, to: MediaPoolItem) {
        guard let fromIndex = filteredItems.firstIndex(where: { $0.id == from.id }),
              let toIndex = filteredItems.firstIndex(where: { $0.id == to.id }) else { return }
        
        let range = min(fromIndex, toIndex)...max(fromIndex, toIndex)
        let rangeItems = filteredItems[range].map { $0.id }
        selectedItems.formUnion(rangeItems)
    }
    
    // Lock Methods (for collaboration)
    func lockItem(_ item: MediaPoolItem) -> Bool {
        guard !lockedItems.contains(item.id) else { return false }
        
        lockedItems.insert(item.id)
        broadcastChange(.locked, for: item)
        return true
    }
    
    func unlockItem(_ item: MediaPoolItem) {
        lockedItems.remove(item.id)
        broadcastChange(.unlocked, for: item)
    }
    
    func isLocked(_ item: MediaPoolItem) -> Bool {
        lockedItems.contains(item.id)
    }
    
    // Refresh Methods
    func refresh() {
        Task {
            isLoading = true
            defer { isLoading = false }
            
            await loadMediaItems()
            await syncWithBackend()
        }
    }
    
    func clearCache() async {
        await thumbnailCache.clearCache()
        itemsCache.removeAllObjects()
        thumbnailCacheSize = 0
    }
    
    // Export Methods
    func exportSelectedItems(to url: URL) async throws {
        let itemsToExport = items.filter { selectedItems.contains($0.id) }
        
        for item in itemsToExport {
            let destination = url.appendingPathComponent(item.url.lastPathComponent)
            try FileManager.default.copyItem(at: item.url, to: destination)
        }
    }
    
    // MARK: - Private Methods
    
    private func loadMediaItems() {
        databaseQueue.async { [weak self] in
            guard let context = self?.persistentContainer?.viewContext else { return }
            
            let request = NSFetchRequest<NSManagedObject>(entityName: "MediaItem")
            
            do {
                let results = try context.fetch(request)
                let loadedItems = results.compactMap { managedObject -> MediaPoolItem? in
                    guard let urlString = managedObject.value(forKey: "url") as? String,
                          let url = URL(string: urlString) else { return nil }
                    return MediaPoolItem(url: url)
                }
                
                Task { @MainActor in
                    self?.items = loadedItems
                    self?.itemCount = loadedItems.count
                    self?.applyFilters()
                }
            } catch {
                Task { @MainActor in
                    self?.error = .databaseError(error.localizedDescription)
                }
            }
        }
    }
    
    private func applyFilters() {
        var filtered = items
        
        // Apply search filter
        if !searchText.isEmpty {
            filtered = filtered.filter { item in
                item.name.localizedCaseInsensitiveContains(searchText) ||
                item.url.lastPathComponent.localizedCaseInsensitiveContains(searchText)
            }
        }
        
        // Apply type filter
        switch filterType {
        case .all:
            break
        case .video:
            filtered = filtered.filter { $0.hasVideo }
        case .audio:
            filtered = filtered.filter { $0.hasAudio && !$0.hasVideo }
        case .images:
            filtered = filtered.filter { !$0.hasVideo && !$0.hasAudio }
        case .recent:
            let dayAgo = Date().addingTimeInterval(-86400)
            filtered = filtered.filter { $0.dateModified > dayAgo }
        case .unused:
            // TODO: Track usage in timeline
            break
        case .inTimeline:
            // TODO: Track timeline usage
            break
        }
        
        filteredItems = filtered
        applySorting()
    }
    
    private func applySorting() {
        switch sortOrder {
        case .name:
            filteredItems.sort { $0.name < $1.name }
        case .dateAdded:
            filteredItems.sort { $0.dateCreated > $1.dateCreated }
        case .dateModified:
            filteredItems.sort { $0.dateModified > $1.dateModified }
        case .size:
            filteredItems.sort { $0.fileSize > $1.fileSize }
        case .duration:
            filteredItems.sort { $0.duration > $1.duration }
        case .type:
            filteredItems.sort { $0.url.pathExtension < $1.url.pathExtension }
        }
    }
    
    private func createMediaItem(from url: URL) async throws -> MediaPoolItem {
        let item = MediaPoolItem(url: url)
        await item.loadMediaProperties()
        await item.generateThumbnail()
        return item
    }
    
    // MARK: - Database Operations
    
    private func saveToDatabase(_ item: MediaPoolItem) {
        databaseQueue.async { [weak self] in
            guard let context = self?.persistentContainer?.viewContext else { return }
            
            let entity = NSEntityDescription.entity(forEntityName: "MediaItem", in: context)!
            let mediaItem = NSManagedObject(entity: entity, insertInto: context)
            
            mediaItem.setValue(item.id.uuidString, forKey: "id")
            mediaItem.setValue(item.url.absoluteString, forKey: "url")
            mediaItem.setValue(item.name, forKey: "name")
            mediaItem.setValue(item.dateCreated, forKey: "dateCreated")
            mediaItem.setValue(item.dateModified, forKey: "dateModified")
            mediaItem.setValue(item.fileSize, forKey: "fileSize")
            mediaItem.setValue(item.duration ?? 0, forKey: "duration")
            
            do {
                try context.save()
            } catch {
                print("Failed to save media item: \(error)")
            }
        }
    }
    
    private func deleteFromDatabase(_ item: MediaPoolItem) {
        databaseQueue.async { [weak self] in
            guard let context = self?.persistentContainer?.viewContext else { return }
            
            let request = NSFetchRequest<NSManagedObject>(entityName: "MediaItem")
            request.predicate = NSPredicate(format: "id == %@", item.id.uuidString)
            
            do {
                let results = try context.fetch(request)
                for object in results {
                    context.delete(object)
                }
                try context.save()
            } catch {
                print("Failed to delete media item: \(error)")
            }
        }
    }
    
    // MARK: - Real-time Sync
    
    private func syncWithBackend() async {
        do {
            // Sync media items with backend
            let response = try await backendService.syncMediaPool(items: items.map { item in
                BackendService.MediaItemData(
                    id: item.id.uuidString,
                    url: item.url.absoluteString,
                    name: item.name,
                    type: item.hasVideo ? "video" : (item.hasAudio ? "audio" : "image"),
                    duration: item.duration ?? 0
                )
            })
            
            // Handle sync response
            // Response is a struct, not a dictionary
            // For now, just log the sync was successful
            print("Media pool synced: \(response.syncedCount) items")
        } catch {
            self.error = .syncFailed(error.localizedDescription)
        }
    }
    
    private func broadcastChange(_ type: MediaChange.ChangeType, for item: MediaPoolItem) {
        let change = MediaChange(
            itemId: item.id,
            userId: ProcessInfo.processInfo.userName,
            changeType: type,
            timestamp: Date(),
            description: "\(type) \(item.name)"
        )
        
        recentChanges.append(change)
        
        // Keep only last 100 changes
        if recentChanges.count > 100 {
            recentChanges.removeFirst()
        }
        
        // Send via WebSocket
        sendWebSocketMessage(change)
    }
    
    private func broadcastSelectionChange(_ selection: Set<UUID>) {
        // Send selection change to other users
        let message: [String: Any] = [
            "type": "selection",
            "userId": ProcessInfo.processInfo.userName,
            "selection": selection.map { $0.uuidString }
        ]
        
        sendWebSocketMessage(message)
    }
    
    private func sendWebSocketMessage(_ message: Any) {
        guard let data = try? JSONSerialization.data(withJSONObject: message) else { return }
        
        syncWebSocket?.send(.data(data)) { error in
            if let error = error {
                print("WebSocket send error: \(error)")
            }
        }
    }
    
    private func receiveWebSocketMessage() {
        syncWebSocket?.receive { [weak self] result in
            switch result {
            case .success(let message):
                switch message {
                case .data(let data):
                    self?.handleWebSocketData(data)
                case .string(let string):
                    if let data = string.data(using: .utf8) {
                        self?.handleWebSocketData(data)
                    }
                @unknown default:
                    break
                }
                
                // Continue receiving
                self?.receiveWebSocketMessage()
                
            case .failure(let error):
                print("WebSocket receive error: \(error)")
                // Attempt reconnection
                Task {
                    try? await Task.sleep(nanoseconds: 5_000_000_000) // 5 seconds
                    self?.setupRealTimeSync()
                }
            }
        }
    }
    
    private func handleWebSocketData(_ data: Data) {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else { return }
        
        Task { @MainActor in
            switch type {
            case "selection":
                handleRemoteSelection(json)
            case "change":
                handleRemoteChange(json)
            case "user":
                handleUserUpdate(json)
            default:
                break
            }
        }
    }
    
    private func handleRemoteSelection(_ data: [String: Any]) {
        guard let userId = data["userId"] as? String,
              let selectionStrings = data["selection"] as? [String] else { return }
        
        let selection = selectionStrings.compactMap { UUID(uuidString: $0) }
        
        // Update collaborating user's selection
        if let userIndex = activeUsers.firstIndex(where: { $0.username == userId }) {
            var user = activeUsers[userIndex]
            activeUsers[userIndex] = CollaboratingUser(
                username: user.username,
                avatar: user.avatar,
                color: user.color,
                currentSelection: selection.first,
                lastActivity: Date(),
                isActive: true
            )
        }
    }
    
    private func handleRemoteChange(_ data: [String: Any]) {
        // Handle changes from other users
        // This would update the media pool based on remote changes
    }
    
    private func handleUserUpdate(_ data: [String: Any]) {
        // Handle user join/leave events
    }
    
    private func handleFileSystemChanges(_ changes: [URL]) {
        Task {
            for url in changes {
                if let existingItem = items.first(where: { $0.url == url }) {
                    // Update existing item
                    await existingItem.loadMediaProperties()
                    await existingItem.generateThumbnail()
                }
            }
        }
    }
}

// MARK: - Supporting Types

class FileSystemWatcher {
    var onChange: (([URL]) -> Void)?
    private var streamRef: FSEventStreamRef?
    
    func startWatching(paths: [String]) {
        let callback: FSEventStreamCallback = { _, _, numEvents, eventPaths, _, _ in
            let paths = Unmanaged<CFArray>.fromOpaque(eventPaths).takeUnretainedValue() as! [String]
            let urls = paths.map { URL(fileURLWithPath: $0) }
            DispatchQueue.main.async {
                // Call onChange handler
            }
        }
        
        var context = FSEventStreamContext()
        context.info = Unmanaged.passUnretained(self).toOpaque()
        
        let pathsToWatch = paths as CFArray
        streamRef = FSEventStreamCreate(
            nil,
            callback,
            &context,
            pathsToWatch,
            FSEventStreamEventId(kFSEventStreamEventIdSinceNow),
            1.0,
            FSEventStreamCreateFlags(kFSEventStreamCreateFlagFileEvents)
        )
        
        if let stream = streamRef {
            FSEventStreamScheduleWithRunLoop(stream, CFRunLoopGetCurrent(), CFRunLoopMode.defaultMode.rawValue)
            FSEventStreamStart(stream)
        }
    }
    
    func stopWatching() {
        if let stream = streamRef {
            FSEventStreamStop(stream)
            FSEventStreamInvalidate(stream)
            FSEventStreamRelease(stream)
            streamRef = nil
        }
    }
    
    deinit {
        stopWatching()
    }
}

class Debouncer {
    private let delay: TimeInterval
    private var workItem: DispatchWorkItem?
    
    init(delay: TimeInterval) {
        self.delay = delay
    }
    
    func debounce(action: @escaping () -> Void) {
        workItem?.cancel()
        let newWorkItem = DispatchWorkItem(block: action)
        workItem = newWorkItem
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: newWorkItem)
    }
}
