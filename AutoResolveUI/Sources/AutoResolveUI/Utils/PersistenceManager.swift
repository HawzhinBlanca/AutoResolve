import Foundation
import SwiftUI

public class PersistenceManager: ObservableObject {
    private let userDefaults = UserDefaults.standard
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    
    // Keys for persistent settings
    private enum Keys {
        static let currentPage = "autoresolve.currentPage"
        static let zoomLevel = "autoresolve.zoomLevel"
        static let scrollOffset = "autoresolve.scrollOffset"
        static let showSilence = "autoresolve.showSilence"
        static let showTranscription = "autoresolve.showTranscription"
        static let showStoryBeats = "autoresolve.showStoryBeats"
        static let showBRoll = "autoresolve.showBRoll"
        static let snapSettings = "autoresolve.snapSettings"
        static let lastProjectPath = "autoresolve.lastProjectPath"
        static let autoProcessOnImport = "autoresolve.autoProcessOnImport"
        static let accessibilitySettings = "autoresolve.accessibilitySettings"
    }
    
    public func saveAppState(_ appState: AppState) {
        userDefaults.set(appState.currentPage.rawValue, forKey: Keys.currentPage)
        userDefaults.set(appState.zoomLevel, forKey: Keys.zoomLevel)
        userDefaults.set(appState.scrollOffset, forKey: Keys.scrollOffset)
        userDefaults.set(appState.showSilence, forKey: Keys.showSilence)
        userDefaults.set(appState.showTranscription, forKey: Keys.showTranscription)
        userDefaults.set(appState.showStoryBeats, forKey: Keys.showStoryBeats)
        userDefaults.set(appState.showBRoll, forKey: Keys.showBRoll)
        userDefaults.set(appState.autoProcessOnImport, forKey: Keys.autoProcessOnImport)
        
        // Save snap settings
        if let data = try? encoder.encode(appState.snapSettings) {
            userDefaults.set(data, forKey: Keys.snapSettings)
        }
        
        // Save last video path
        if let videoURL = appState.videoURL {
            userDefaults.set(videoURL.path, forKey: Keys.lastProjectPath)
        }
    }
    
    public func restoreAppState(_ appState: AppState) {
        // Restore UI settings
        if let pageString = userDefaults.object(forKey: Keys.currentPage) as? String,
           let page = AppState.Page(rawValue: pageString) {
            appState.currentPage = page
        }
        
        if userDefaults.object(forKey: Keys.zoomLevel) != nil {
            appState.zoomLevel = userDefaults.double(forKey: Keys.zoomLevel)
            if appState.zoomLevel <= 0 { appState.zoomLevel = 1.0 }
        }
        
        if userDefaults.object(forKey: Keys.scrollOffset) != nil {
            appState.scrollOffset = CGFloat(userDefaults.double(forKey: Keys.scrollOffset))
        }
        
        // Restore lane visibility
        if userDefaults.object(forKey: Keys.showSilence) != nil {
            appState.showSilence = userDefaults.bool(forKey: Keys.showSilence)
        }
        if userDefaults.object(forKey: Keys.showTranscription) != nil {
            appState.showTranscription = userDefaults.bool(forKey: Keys.showTranscription)
        }
        if userDefaults.object(forKey: Keys.showStoryBeats) != nil {
            appState.showStoryBeats = userDefaults.bool(forKey: Keys.showStoryBeats)
        }
        if userDefaults.object(forKey: Keys.showBRoll) != nil {
            appState.showBRoll = userDefaults.bool(forKey: Keys.showBRoll)
        }
        
        if userDefaults.object(forKey: Keys.autoProcessOnImport) != nil {
            appState.autoProcessOnImport = userDefaults.bool(forKey: Keys.autoProcessOnImport)
        }
        
        // Restore snap settings
        if let data = userDefaults.data(forKey: Keys.snapSettings),
           let snapSettings = try? decoder.decode(SnapSettings.self, from: data) {
            appState.snapSettings = snapSettings
        }
        
        // Restore last project
        if let lastPath = userDefaults.string(forKey: Keys.lastProjectPath),
           FileManager.default.fileExists(atPath: lastPath) {
            appState.videoURL = URL(fileURLWithPath: lastPath)
        }
    }
    
    public func saveProject(_ appState: AppState, to url: URL) throws {
        let projectData = ProjectData(
            videoURL: appState.videoURL,
            timeline: appState.timeline,
            zoomLevel: appState.zoomLevel,
            scrollOffset: appState.scrollOffset,
            snapSettings: appState.snapSettings,
            showSilence: appState.showSilence,
            showTranscription: appState.showTranscription,
            showStoryBeats: appState.showStoryBeats,
            showBRoll: appState.showBRoll
        )
        
        let data = try encoder.encode(projectData)
        try data.write(to: url)
    }
    
    public func loadProject(from url: URL, into appState: AppState) throws {
        let data = try Data(contentsOf: url)
        let projectData = try decoder.decode(ProjectData.self, from: data)
        
        appState.videoURL = projectData.videoURL
        appState.timeline = projectData.timeline
        appState.zoomLevel = projectData.zoomLevel
        appState.scrollOffset = projectData.scrollOffset
        appState.snapSettings = projectData.snapSettings
        appState.showSilence = projectData.showSilence
        appState.showTranscription = projectData.showTranscription
        appState.showStoryBeats = projectData.showStoryBeats
        appState.showBRoll = projectData.showBRoll
    }
    
    public func clearCache() {
        let keys = [
            Keys.currentPage,
            Keys.zoomLevel,
            Keys.scrollOffset,
            Keys.showSilence,
            Keys.showTranscription,
            Keys.showStoryBeats,
            Keys.showBRoll,
            Keys.snapSettings,
            Keys.lastProjectPath,
            Keys.autoProcessOnImport,
            Keys.accessibilitySettings
        ]
        
        for key in keys {
            userDefaults.removeObject(forKey: key)
        }
    }
}

struct ProjectData: Codable {
    let videoURL: URL?
    let timeline: TimelineModel?
    let zoomLevel: Double
    let scrollOffset: CGFloat
    let snapSettings: SnapSettings
    let showSilence: Bool
    let showTranscription: Bool
    let showStoryBeats: Bool
    let showBRoll: Bool
}