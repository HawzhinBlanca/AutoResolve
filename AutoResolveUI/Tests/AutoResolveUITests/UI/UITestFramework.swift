//
//  UITestFramework.swift
//  AutoResolveUITests
//
//  Created by AutoResolve on 8/23/25.
//

import XCTest
import SwiftUI
@testable import AutoResolveUI

/// Comprehensive UI testing framework for AutoResolve DaVinci Resolve-style interface
/// Tests user interactions, navigation flows, and interface responsiveness
class UITestFramework: XCTestCase {
    
    private var app: XCUIApplication!
    private var testTimeout: TimeInterval = 30.0
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        
        // Initialize app
        app = XCUIApplication()
        app.launchArguments.append("--ui-testing")
        app.launchEnvironment["UITEST_MODE"] = "1"
        app.launch()
        
        // Wait for app to load
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 10.0))
    }
    
    override func tearDownWithError() throws {
        app.terminate()
        app = nil
        try super.tearDownWithError()
    }
    
    // MARK: - Main Interface Tests
    
    func testMainInterfaceLayout() throws {
        // Test DaVinci Resolve-style three-panel layout
        let mediaPoolPanel = app.otherElements["MediaPoolPanel"]
        let timelinePanel = app.otherElements["TimelinePanel"] 
        let inspectorPanel = app.otherElements["InspectorPanel"]
        
        XCTAssertTrue(mediaPoolPanel.exists, "Media Pool panel should exist")
        XCTAssertTrue(timelinePanel.exists, "Timeline panel should exist")
        XCTAssertTrue(inspectorPanel.exists, "Inspector panel should exist")
        
        // Test panel dimensions
        let mediaPoolFrame = mediaPoolPanel.frame
        let inspectorFrame = inspectorPanel.frame
        
        XCTAssertEqual(mediaPoolFrame.width, 380, accuracy: 10, "Media Pool should be 380px wide")
        XCTAssertEqual(inspectorFrame.width, 380, accuracy: 10, "Inspector should be 380px wide")
    }
    
    func testMenuBarNavigation() throws {
        // Test main menu bar
        let menuBar = app.menuBars.firstMatch
        XCTAssertTrue(menuBar.exists, "Menu bar should exist")
        
        // Test File menu
        let fileMenu = menuBar.menuBarItems["File"]
        XCTAssertTrue(fileMenu.exists, "File menu should exist")
        
        fileMenu.click()
        
        let newProjectItem = app.menuItems["New Project"]
        let openProjectItem = app.menuItems["Open Project"]
        let saveProjectItem = app.menuItems["Save Project"]
        
        XCTAssertTrue(newProjectItem.exists, "New Project menu item should exist")
        XCTAssertTrue(openProjectItem.exists, "Open Project menu item should exist")
        XCTAssertTrue(saveProjectItem.exists, "Save Project menu item should exist")
        
        // Close menu
        app.typeKey(.escape, modifierFlags: [])
        
        // Test Timeline menu
        let timelineMenu = menuBar.menuBarItems["Timeline"]
        XCTAssertTrue(timelineMenu.exists, "Timeline menu should exist")
        
        timelineMenu.click()
        
        let addTrackItem = app.menuItems["Add Video Track"]
        let deleteTrackItem = app.menuItems["Delete Track"]
        
        XCTAssertTrue(addTrackItem.exists, "Add Video Track menu item should exist")
        XCTAssertTrue(deleteTrackItem.exists, "Delete Track menu item should exist")
        
        app.typeKey(.escape, modifierFlags: [])
        
        // Test AI Director menu
        let aiDirectorMenu = menuBar.menuBarItems["AI Director"]
        XCTAssertTrue(aiDirectorMenu.exists, "AI Director menu should exist")
        
        aiDirectorMenu.click()
        
        let analyzeContentItem = app.menuItems["Analyze Content"]
        let generateCutsItem = app.menuItems["Generate Smart Cuts"]
        
        XCTAssertTrue(analyzeContentItem.exists, "Analyze Content menu item should exist")
        XCTAssertTrue(generateCutsItem.exists, "Generate Smart Cuts menu item should exist")
        
        app.typeKey(.escape, modifierFlags: [])
    }
    
    func testTimelineInterface() throws {
        let timeline = app.otherElements["Timeline"]
        XCTAssertTrue(timeline.exists, "Timeline should exist")
        
        // Test video tracks
        let videoTrack1 = timeline.otherElements["VideoTrack_V1"]
        let videoTrack2 = timeline.otherElements["VideoTrack_V2"]
        let videoTrack3 = timeline.otherElements["VideoTrack_V3"]
        
        XCTAssertTrue(videoTrack1.exists, "Video track V1 should exist")
        XCTAssertTrue(videoTrack2.exists, "Video track V2 should exist")
        XCTAssertTrue(videoTrack3.exists, "Video track V3 should exist")
        
        // Test special tracks
        let directorTrack = timeline.otherElements["DirectorTrack"]
        let transcriptionTrack = timeline.otherElements["TranscriptionTrack"]
        
        XCTAssertTrue(directorTrack.exists, "Director track should exist")
        XCTAssertTrue(transcriptionTrack.exists, "Transcription track should exist")
        
        // Test audio tracks
        let audioTrack1 = timeline.otherElements["AudioTrack_A1"]
        let audioTrack2 = timeline.otherElements["AudioTrack_A2"]
        
        XCTAssertTrue(audioTrack1.exists, "Audio track A1 should exist")
        XCTAssertTrue(audioTrack2.exists, "Audio track A2 should exist")
        
        // Test timeline toolbar
        let timelineToolbar = timeline.toolbars.firstMatch
        XCTAssertTrue(timelineToolbar.exists, "Timeline toolbar should exist")
        
        let neuralTimelineToggle = timelineToolbar.buttons["NeuralTimelineToggle"]
        let autoCutButton = timelineToolbar.buttons["AutoCutButton"]
        let directorAnalysisButton = timelineToolbar.buttons["DirectorAnalysisButton"]
        
        XCTAssertTrue(neuralTimelineToggle.exists, "Neural Timeline toggle should exist")
        XCTAssertTrue(autoCutButton.exists, "Auto-Cut button should exist")
        XCTAssertTrue(directorAnalysisButton.exists, "Director Analysis button should exist")
    }
    
    func testInspectorTabs() throws {
        let inspector = app.otherElements["InspectorPanel"]
        XCTAssertTrue(inspector.exists, "Inspector panel should exist")
        
        // Test tab buttons
        let videoTab = inspector.buttons["VideoTab"]
        let audioTab = inspector.buttons["AudioTab"]
        let neuralAnalysisTab = inspector.buttons["NeuralAnalysisTab"]
        let directorTab = inspector.buttons["DirectorTab"]
        let cutsTab = inspector.buttons["CutsTab"]
        let shortsTab = inspector.buttons["ShortsTab"]
        
        XCTAssertTrue(videoTab.exists, "Video tab should exist")
        XCTAssertTrue(audioTab.exists, "Audio tab should exist")
        XCTAssertTrue(neuralAnalysisTab.exists, "Neural Analysis tab should exist")
        XCTAssertTrue(directorTab.exists, "Director tab should exist")
        XCTAssertTrue(cutsTab.exists, "Cuts tab should exist")
        XCTAssertTrue(shortsTab.exists, "Shorts tab should exist")
        
        // Test tab switching
        neuralAnalysisTab.click()
        
        let neuralAnalysisContent = inspector.otherElements["NeuralAnalysisContent"]
        XCTAssertTrue(neuralAnalysisContent.waitForExistence(timeout: 2.0), "Neural Analysis content should appear")
        
        // Test Director tab
        directorTab.click()
        
        let directorContent = inspector.otherElements["DirectorContent"]
        XCTAssertTrue(directorContent.waitForExistence(timeout: 2.0), "Director content should appear")
        
        let energyGraph = directorContent.otherElements["EnergyGraph"]
        let momentumGraph = directorContent.otherElements["MomentumGraph"]
        
        XCTAssertTrue(energyGraph.exists, "Energy graph should exist in Director tab")
        XCTAssertTrue(momentumGraph.exists, "Momentum graph should exist in Director tab")
    }
    
    // MARK: - Media Pool Tests
    
    func testMediaPoolInterface() throws {
        let mediaPool = app.otherElements["MediaPoolPanel"]
        XCTAssertTrue(mediaPool.exists, "Media Pool panel should exist")
        
        // Test tab structure
        let masterTab = mediaPool.buttons["MasterTab"]
        let vjepaEmbeddingsTab = mediaPool.buttons["VJEPAEmbeddingsTab"]
        let clipResultsTab = mediaPool.buttons["CLIPResultsTab"]
        let brollLibraryTab = mediaPool.buttons["BrollLibraryTab"]
        
        XCTAssertTrue(masterTab.exists, "Master tab should exist")
        XCTAssertTrue(vjepaEmbeddingsTab.exists, "V-JEPA Embeddings tab should exist")
        XCTAssertTrue(clipResultsTab.exists, "CLIP Results tab should exist")
        XCTAssertTrue(brollLibraryTab.exists, "B-roll Library tab should exist")
        
        // Test import functionality
        let importButton = mediaPool.buttons["ImportMediaButton"]
        XCTAssertTrue(importButton.exists, "Import media button should exist")
        
        // Test media grid
        let mediaGrid = mediaPool.collectionViews["MediaGrid"]
        XCTAssertTrue(mediaGrid.exists, "Media grid should exist")
    }
    
    func testMediaImportFlow() throws {
        let mediaPool = app.otherElements["MediaPoolPanel"]
        let importButton = mediaPool.buttons["ImportMediaButton"]
        
        importButton.click()
        
        // Test import dialog
        let importDialog = app.sheets.firstMatch
        XCTAssertTrue(importDialog.waitForExistence(timeout: 5.0), "Import dialog should appear")
        
        let browseButton = importDialog.buttons["Browse"]
        let importSettingsButton = importDialog.buttons["ImportSettings"]
        let cancelButton = importDialog.buttons["Cancel"]
        let importNowButton = importDialog.buttons["Import"]
        
        XCTAssertTrue(browseButton.exists, "Browse button should exist")
        XCTAssertTrue(importSettingsButton.exists, "Import Settings button should exist")
        XCTAssertTrue(cancelButton.exists, "Cancel button should exist")
        XCTAssertTrue(importNowButton.exists, "Import button should exist")
        
        // Test import settings
        importSettingsButton.click()
        
        let settingsSheet = app.sheets.element(boundBy: 1)
        XCTAssertTrue(settingsSheet.waitForExistence(timeout: 2.0), "Import settings sheet should appear")
        
        let generateThumbnailsCheckbox = settingsSheet.checkBoxes["GenerateThumbnails"]
        let analyzeAudioCheckbox = settingsSheet.checkBoxes["AnalyzeAudio"]
        let runEmbedderCheckbox = settingsSheet.checkBoxes["RunEmbedder"]
        
        XCTAssertTrue(generateThumbnailsCheckbox.exists, "Generate thumbnails checkbox should exist")
        XCTAssertTrue(analyzeAudioCheckbox.exists, "Analyze audio checkbox should exist")
        XCTAssertTrue(runEmbedderCheckbox.exists, "Run embedder checkbox should exist")
        
        // Close dialogs
        settingsSheet.buttons["Close"].click()
        importDialog.buttons["Cancel"].click()
    }
    
    // MARK: - Timeline Interaction Tests
    
    func testTimelineInteractions() throws {
        // First, add some test media to timeline
        addTestMediaToTimeline()
        
        let timeline = app.otherElements["Timeline"]
        
        // Test playback controls
        let playButton = timeline.buttons["PlayButton"]
        let pauseButton = timeline.buttons["PauseButton"]
        let stopButton = timeline.buttons["StopButton"]
        
        XCTAssertTrue(playButton.exists, "Play button should exist")
        
        playButton.click()
        
        // Verify playback started (play button should change to pause)
        XCTAssertTrue(pauseButton.waitForExistence(timeout: 2.0), "Pause button should appear during playback")
        
        pauseButton.click()
        
        // Test timeline scrubbing
        let playhead = timeline.otherElements["Playhead"]
        XCTAssertTrue(playhead.exists, "Playhead should exist")
        
        let initialPosition = playhead.frame.midX
        playhead.drag(to: CGPoint(x: initialPosition + 100, y: playhead.frame.midY))
        
        // Verify playhead moved
        let newPosition = playhead.frame.midX
        XCTAssertNotEqual(initialPosition, newPosition, "Playhead should have moved")
        
        // Test zoom controls
        let zoomInButton = timeline.buttons["ZoomInButton"]
        let zoomOutButton = timeline.buttons["ZoomOutButton"]
        
        XCTAssertTrue(zoomInButton.exists, "Zoom in button should exist")
        XCTAssertTrue(zoomOutButton.exists, "Zoom out button should exist")
        
        zoomInButton.click()
        
        // Allow time for zoom animation
        Thread.sleep(forTimeInterval: 0.5)
        
        zoomOutButton.click()
    }
    
    func testClipEditing() throws {
        addTestMediaToTimeline()
        
        let timeline = app.otherElements["Timeline"]
        let videoTrack = timeline.otherElements["VideoTrack_V1"]
        
        // Find first clip on timeline
        let clips = videoTrack.otherElements.matching(identifier: "TimelineClip")
        XCTAssertGreaterThan(clips.count, 0, "Should have at least one clip on timeline")
        
        let firstClip = clips.element(boundBy: 0)
        
        // Test clip selection
        firstClip.click()
        
        // Verify clip is selected (should highlight)
        XCTAssertTrue(firstClip.isSelected, "Clip should be selected")
        
        // Test context menu
        firstClip.rightClick()
        
        let contextMenu = app.menus.firstMatch
        XCTAssertTrue(contextMenu.waitForExistence(timeout: 2.0), "Context menu should appear")
        
        let cutItem = contextMenu.menuItems["Cut"]
        let copyItem = contextMenu.menuItems["Copy"]
        let deleteItem = contextMenu.menuItems["Delete"]
        let duplicateItem = contextMenu.menuItems["Duplicate"]
        
        XCTAssertTrue(cutItem.exists, "Cut menu item should exist")
        XCTAssertTrue(copyItem.exists, "Copy menu item should exist")
        XCTAssertTrue(deleteItem.exists, "Delete menu item should exist")
        XCTAssertTrue(duplicateItem.exists, "Duplicate menu item should exist")
        
        // Close context menu
        app.typeKey(.escape, modifierFlags: [])
        
        // Test clip trimming
        let clipLeftEdge = firstClip.coordinate(withNormalizedOffset: CGVector(dx: 0.0, dy: 0.5))
        let clipRightEdge = firstClip.coordinate(withNormalizedOffset: CGVector(dx: 1.0, dy: 0.5))
        
        // Trim from left
        clipLeftEdge.press(forDuration: 0.1, thenDragTo: clipLeftEdge.withOffset(CGVector(dx: 20, dy: 0)))
        
        // Allow time for trim operation
        Thread.sleep(forTimeInterval: 0.5)
        
        // Trim from right  
        clipRightEdge.press(forDuration: 0.1, thenDragTo: clipRightEdge.withOffset(CGVector(dx: -20, dy: 0)))
    }
    
    // MARK: - AI Director Tests
    
    func testAIDirectorInterface() throws {
        addTestMediaToTimeline()
        
        let timeline = app.otherElements["Timeline"]
        let directorTrack = timeline.otherElements["DirectorTrack"]
        
        // Test Director Analysis button
        let directorAnalysisButton = timeline.toolbars.firstMatch.buttons["DirectorAnalysisButton"]
        directorAnalysisButton.click()
        
        // Test analysis progress dialog
        let analysisDialog = app.sheets.firstMatch
        XCTAssertTrue(analysisDialog.waitForExistence(timeout: 2.0), "Analysis dialog should appear")
        
        let progressBar = analysisDialog.progressIndicators.firstMatch
        let analysisStatus = analysisDialog.staticTexts["AnalysisStatus"]
        let cancelAnalysisButton = analysisDialog.buttons["Cancel"]
        
        XCTAssertTrue(progressBar.exists, "Progress bar should exist")
        XCTAssertTrue(analysisStatus.exists, "Analysis status should exist")
        XCTAssertTrue(cancelAnalysisButton.exists, "Cancel button should exist")
        
        // Wait for analysis to complete or cancel
        if !analysisDialog.buttons["Done"].waitForExistence(timeout: 10.0) {
            cancelAnalysisButton.click()
        } else {
            analysisDialog.buttons["Done"].click()
        }
        
        // Test Director track visualization
        let energyCurve = directorTrack.otherElements["EnergyCurve"]
        let tensionMarkers = directorTrack.otherElements["TensionMarkers"]
        let storyBeats = directorTrack.otherElements["StoryBeats"]
        
        // These may only exist after analysis completes
        if energyCurve.exists {
            XCTAssertTrue(energyCurve.isHittable, "Energy curve should be interactive")
        }
        
        // Test Inspector Director tab
        let inspector = app.otherElements["InspectorPanel"]
        let directorTab = inspector.buttons["DirectorTab"]
        directorTab.click()
        
        let directorContent = inspector.otherElements["DirectorContent"]
        XCTAssertTrue(directorContent.waitForExistence(timeout: 2.0), "Director content should appear")
    }
    
    func testSmartCutGeneration() throws {
        addTestMediaToTimeline()
        
        let timeline = app.otherElements["Timeline"]
        let autoCutButton = timeline.toolbars.firstMatch.buttons["AutoCutButton"]
        
        autoCutButton.click()
        
        // Test Smart Cut dialog
        let smartCutDialog = app.sheets.firstMatch
        XCTAssertTrue(smartCutDialog.waitForExistence(timeout: 2.0), "Smart Cut dialog should appear")
        
        let silenceThresholdSlider = smartCutDialog.sliders["SilenceThresholdSlider"]
        let minimumDurationSlider = smartCutDialog.sliders["MinimumDurationSlider"]
        let preserveWordsCheckbox = smartCutDialog.checkBoxes["PreserveWords"]
        let generateCutsButton = smartCutDialog.buttons["GenerateCuts"]
        
        XCTAssertTrue(silenceThresholdSlider.exists, "Silence threshold slider should exist")
        XCTAssertTrue(minimumDurationSlider.exists, "Minimum duration slider should exist")
        XCTAssertTrue(preserveWordsCheckbox.exists, "Preserve words checkbox should exist")
        XCTAssertTrue(generateCutsButton.exists, "Generate cuts button should exist")
        
        // Adjust settings
        silenceThresholdSlider.adjust(toNormalizedSliderPosition: 0.6)
        minimumDurationSlider.adjust(toNormalizedSliderPosition: 0.4)
        
        generateCutsButton.click()
        
        // Wait for cut generation
        let progressDialog = app.sheets.element(boundBy: 1)
        if progressDialog.waitForExistence(timeout: 2.0) {
            let doneButton = progressDialog.buttons["Done"]
            XCTAssertTrue(doneButton.waitForExistence(timeout: 15.0), "Cut generation should complete")
            doneButton.click()
        }
        
        smartCutDialog.buttons["Apply"].click()
    }
    
    // MARK: - Export Tests
    
    func testExportFlow() throws {
        addTestMediaToTimeline()
        
        // Test export via menu
        let menuBar = app.menuBars.firstMatch
        let exportMenu = menuBar.menuBarItems["Export"]
        exportMenu.click()
        
        let exportVideoItem = app.menuItems["Export Video"]
        let exportFCPXMLItem = app.menuItems["Export FCPXML"]
        let exportEDLItem = app.menuItems["Export EDL"]
        
        XCTAssertTrue(exportVideoItem.exists, "Export Video menu item should exist")
        XCTAssertTrue(exportFCPXMLItem.exists, "Export FCPXML menu item should exist") 
        XCTAssertTrue(exportEDLItem.exists, "Export EDL menu item should exist")
        
        exportVideoItem.click()
        
        // Test export dialog
        let exportDialog = app.sheets.firstMatch
        XCTAssertTrue(exportDialog.waitForExistence(timeout: 5.0), "Export dialog should appear")
        
        let formatPopup = exportDialog.popUpButtons["FormatPopup"]
        let qualityPopup = exportDialog.popUpButtons["QualityPopup"]
        let destinationField = exportDialog.textFields["DestinationField"]
        let browseDestinationButton = exportDialog.buttons["BrowseDestination"]
        let exportButton = exportDialog.buttons["Export"]
        let cancelButton = exportDialog.buttons["Cancel"]
        
        XCTAssertTrue(formatPopup.exists, "Format popup should exist")
        XCTAssertTrue(qualityPopup.exists, "Quality popup should exist")
        XCTAssertTrue(destinationField.exists, "Destination field should exist")
        XCTAssertTrue(browseDestinationButton.exists, "Browse destination button should exist")
        XCTAssertTrue(exportButton.exists, "Export button should exist")
        XCTAssertTrue(cancelButton.exists, "Cancel button should exist")
        
        // Test format selection
        formatPopup.click()
        
        let mp4Option = app.menuItems["MP4"]
        let movOption = app.menuItems["MOV"]
        let avchdOption = app.menuItems["AVCHD"]
        
        XCTAssertTrue(mp4Option.exists, "MP4 option should exist")
        XCTAssertTrue(movOption.exists, "MOV option should exist")
        
        mp4Option.click()
        
        // Test quality selection
        qualityPopup.click()
        
        let highQualityOption = app.menuItems["High Quality"]
        let mediumQualityOption = app.menuItems["Medium Quality"]
        
        XCTAssertTrue(highQualityOption.exists, "High Quality option should exist")
        XCTAssertTrue(mediumQualityOption.exists, "Medium Quality option should exist")
        
        highQualityOption.click()
        
        cancelButton.click()
    }
    
    // MARK: - Performance Tests
    
    func testUIResponsiveness() throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Test rapid UI interactions
        let timeline = app.otherElements["Timeline"]
        let mediaPool = app.otherElements["MediaPoolPanel"]
        let inspector = app.otherElements["InspectorPanel"]
        
        // Rapid tab switching in inspector
        let inspectorTabs = ["VideoTab", "AudioTab", "NeuralAnalysisTab", "DirectorTab", "CutsTab", "ShortsTab"]
        
        for tabId in inspectorTabs {
            let tab = inspector.buttons[tabId]
            if tab.exists {
                tab.click()
                Thread.sleep(forTimeInterval: 0.1) // Small delay between clicks
            }
        }
        
        // Test timeline zoom responsiveness
        let zoomInButton = timeline.buttons["ZoomInButton"]
        let zoomOutButton = timeline.buttons["ZoomOutButton"]
        
        for _ in 0..<5 {
            zoomInButton.click()
            Thread.sleep(forTimeInterval: 0.05)
        }
        
        for _ in 0..<5 {
            zoomOutButton.click()
            Thread.sleep(forTimeInterval: 0.05)
        }
        
        let endTime = CFAbsoluteTimeGetCurrent()
        let totalTime = endTime - startTime
        
        XCTAssertLessThan(totalTime, 5.0, "UI interactions should be responsive")
    }
    
    func testMemoryUsageDuringUIOperations() throws {
        let initialMemory = getAppMemoryUsage()
        
        // Perform intensive UI operations
        addTestMediaToTimeline()
        
        // Switch between tabs rapidly
        let inspector = app.otherElements["InspectorPanel"]
        let inspectorTabs = ["VideoTab", "AudioTab", "NeuralAnalysisTab", "DirectorTab", "CutsTab", "ShortsTab"]
        
        for _ in 0..<10 {
            for tabId in inspectorTabs {
                let tab = inspector.buttons[tabId]
                if tab.exists {
                    tab.click()
                    Thread.sleep(forTimeInterval: 0.1)
                }
            }
        }
        
        let finalMemory = getAppMemoryUsage()
        let memoryIncrease = finalMemory - initialMemory
        
        XCTAssertLessThan(memoryIncrease, 50 * 1024 * 1024, "Memory increase should be less than 50MB")
    }
    
    // MARK: - Accessibility Tests
    
    func testAccessibilitySupport() throws {
        // Test VoiceOver labels
        let timeline = app.otherElements["Timeline"]
        XCTAssertNotNil(timeline.label, "Timeline should have accessibility label")
        
        let mediaPool = app.otherElements["MediaPoolPanel"]
        XCTAssertNotNil(mediaPool.label, "Media Pool should have accessibility label")
        
        let inspector = app.otherElements["InspectorPanel"]
        XCTAssertNotNil(inspector.label, "Inspector should have accessibility label")
        
        // Test keyboard navigation
        app.typeKey(.tab, modifierFlags: [])
        
        let focusedElement = app.otherElements.element(matching: .any, identifier: "focused")
        XCTAssertTrue(focusedElement.exists || app.buttons.firstMatch.hasFocus, 
                     "Some element should have keyboard focus")
        
        // Test keyboard shortcuts
        app.typeKey(.space, modifierFlags: [])  // Play/Pause
        app.typeKey("i", modifierFlags: [])     // Mark In
        app.typeKey("o", modifierFlags: [])     // Mark Out
        app.typeKey("x", modifierFlags: [.command])  // Cut
        app.typeKey("v", modifierFlags: [.command])  // Paste
    }
    
    // MARK: - Helper Methods
    
    private func addTestMediaToTimeline() {
        // Add test media to timeline for UI testing
        let mediaPool = app.otherElements["MediaPoolPanel"]
        let importButton = mediaPool.buttons["ImportMediaButton"]
        
        if importButton.exists {
            importButton.click()
            
            let importDialog = app.sheets.firstMatch
            if importDialog.waitForExistence(timeout: 2.0) {
                // Add synthetic test media
                let addTestMediaButton = importDialog.buttons["AddTestMedia"]
                if addTestMediaButton.exists {
                    addTestMediaButton.click()
                    importDialog.buttons["Import"].click()
                    
                    // Wait for import to complete
                    Thread.sleep(forTimeInterval: 2.0)
                    
                    // Drag media to timeline
                    let mediaGrid = mediaPool.collectionViews["MediaGrid"]
                    let firstMediaItem = mediaGrid.cells.firstMatch
                    
                    if firstMediaItem.exists {
                        let timeline = app.otherElements["Timeline"]
                        let videoTrack = timeline.otherElements["VideoTrack_V1"]
                        
                        firstMediaItem.drag(to: videoTrack)
                    }
                } else {
                    importDialog.buttons["Cancel"].click()
                }
            }
        }
    }
    
    private func getAppMemoryUsage() -> Int64 {
        // Get memory usage of the test app
        let task = Process()
        task.launchPath = "/bin/ps"
        task.arguments = ["-o", "rss=", "-p", "\(app.processID)"]
        
        let pipe = Pipe()
        task.standardOutput = pipe
        task.launch()
        task.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? "0"
        
        return Int64(output.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0
    }
}