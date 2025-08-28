import Foundation
import SwiftUI
import Combine
import AVFoundation
import Vision
import CoreML
import CoreImage
import OSLog
import CoreGraphics

/// Advanced visual analysis system using Vision framework and CoreML models
/// Performs scene classification, object detection, composition analysis, and visual quality assessment
@MainActor
public class VisualAnalyzer: ObservableObject {
    private let logger = Logger.shared
    
    // Vision models
    private var sceneClassifier: VNClassifyImageRequest?
    private var objectDetector: VNDetectRectanglesRequest?
    private var faceLandmarkDetector: VNDetectFaceLandmarksRequest?
    private var visualQualityAnalyzer: VNClassifyImageRequest?
    
    // CoreML models
    private var customSceneModel: VNCoreMLModel?
    private var compositionModel: VNCoreMLModel?
    private var aestheticsModel: VNCoreMLModel?
    
    // Analysis state
    @Published public var isAnalyzing = false
    @Published public var analysisProgress = 0.0
    @Published public var currentOperation = ""
    
    // Results cache
    private var analysisCache: [String: VisualAnalysis] = [:]
    
    public init() {
        setupVisionModels()
        loadCoreMLModels()
    }
    
    // MARK: - Public API
    
    public func analyzeVideo(_ videoURL: URL) async throws -> VisualAnalysis {
        let cacheKey = videoURL.absoluteString
        if let cached = analysisCache[cacheKey] {
            return cached
        }
        
        logger.info("Starting visual analysis for: \(videoURL.lastPathComponent)")
        
        isAnalyzing = true
        analysisProgress = 0.0
        currentOperation = "Initializing visual analysis..."
        
        defer {
            isAnalyzing = false
            analysisProgress = 1.0
            currentOperation = ""
        }
        
        do {
            let asset = AVAsset(url: videoURL)
            let duration = try await asset.load(.duration)
            let reader = try AVAssetReader(asset: asset)
            
            // Configure video track
            guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
                throw AnalysisError.noVideoTrack
            }
            
            let outputSettings: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: 1920,
                kCVPixelBufferHeightKey as String: 1080
            ]
            
            let output = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: outputSettings)
            reader.add(output)
            reader.startReading()
            
            var frameAnalyses: [FrameAnalysis] = []
            let sampleInterval = CMTime(seconds: 1.0, preferredTimescale: 600) // Analyze every second
            var currentTime = CMTime.zero
            
            while reader.status == .reading {
                if let sampleBuffer = output.copyNextSampleBuffer() {
                    let frameTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                    
                    if frameTime >= currentTime {
                        currentOperation = "Analyzing frame at \(frameTime.seconds)s..."
                        analysisProgress = Double(frameTime.seconds / duration.seconds) * 0.8
                        
                        let frameAnalysis = try await analyzeFrame(sampleBuffer, timestamp: frameTime)
                        frameAnalyses.append(frameAnalysis)
                        
                        currentTime = frameTime + sampleInterval
                    }
                    
                    CMSampleBufferInvalidate(sampleBuffer)
                } else {
                    break
                }
            }
            
            currentOperation = "Aggregating visual analysis results..."
            analysisProgress = 0.9
            
            let visualAnalysis = aggregateFrameAnalyses(frameAnalyses, videoDuration: duration)
            
            // Cache results
            analysisCache[cacheKey] = visualAnalysis
            
            logger.info("Visual analysis completed with \(frameAnalyses.count) frames analyzed")
            return visualAnalysis
            
        } catch {
            logger.error("Visual analysis failed: \(error)")
            throw error
        }
    }
    
    public func analyzeImage(_ image: CGImage) async throws -> FrameAnalysis {
        logger.info("Starting image analysis")
        
        let request = VNImageRequestHandler(cgImage: image)
        var results: [VNObservation] = []
        
        // Scene classification
        if let sceneClassifier = sceneClassifier {
            try request.perform([sceneClassifier])
            results.append(contentsOf: sceneClassifier.results ?? [])
        }
        
        // Object detection
        if let objectDetector = objectDetector {
            try request.perform([objectDetector])
            results.append(contentsOf: objectDetector.results ?? [])
        }
        
        // Face landmarks
        if let faceLandmarkDetector = faceLandmarkDetector {
            try request.perform([faceLandmarkDetector])
            results.append(contentsOf: faceLandmarkDetector.results ?? [])
        }
        
        return processVisionResults(results, image: image, timestamp: .zero)
    }
    
    // MARK: - Private Implementation
    
    private func setupVisionModels() {
        // Scene classification
        sceneClassifier = VNClassifyImageRequest { [weak self] request, error in
            if let error = error {
                self?.logger.error("Scene classification error: \(error)")
            }
        }
        // Image crop and scale option not available in current Vision API
        
        // Rectangle detection (for composition analysis)
        objectDetector = VNDetectRectanglesRequest { [weak self] request, error in
            if let error = error {
                self?.logger.error("Rectangle detection error: \(error)")
            }
        }
        objectDetector?.minimumAspectRatio = 0.3
        objectDetector?.maximumAspectRatio = 3.0
        objectDetector?.minimumSize = 0.1
        
        // Face landmark detection
        faceLandmarkDetector = VNDetectFaceLandmarksRequest { [weak self] request, error in
            if let error = error {
                self?.logger.error("Face landmark detection error: \(error)")
            }
        }
    }
    
    private func loadCoreMLModels() {
        // Load custom scene classification model
        if let modelURL = Bundle.main.url(forResource: "SceneClassifier", withExtension: "mlmodelc") {
            do {
                let model = try MLModel(contentsOf: modelURL)
                customSceneModel = try VNCoreMLModel(for: model)
            } catch {
                logger.error("Failed to load custom scene model: \(error)")
            }
        }
        
        // Load composition analysis model
        if let modelURL = Bundle.main.url(forResource: "CompositionAnalyzer", withExtension: "mlmodelc") {
            do {
                let model = try MLModel(contentsOf: modelURL)
                compositionModel = try VNCoreMLModel(for: model)
            } catch {
                logger.error("Failed to load composition model: \(error)")
            }
        }
        
        // Load aesthetics model
        if let modelURL = Bundle.main.url(forResource: "AestheticsClassifier", withExtension: "mlmodelc") {
            do {
                let model = try MLModel(contentsOf: modelURL)
                aestheticsModel = try VNCoreMLModel(for: model)
            } catch {
                logger.error("Failed to load aesthetics model: \(error)")
            }
        }
    }
    
    private func analyzeFrame(_ sampleBuffer: CMSampleBuffer, timestamp: CMTime) async throws -> FrameAnalysis {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            throw AnalysisError.invalidSampleBuffer
        }
        
        let ciImage = CIImage(cvImageBuffer: imageBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            throw AnalysisError.imageProcessingFailed
        }
        
        let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        var observations: [VNObservation] = []
        
        // Perform all vision requests
        var requests: [VNRequest] = []
        
        if let sceneClassifier = sceneClassifier {
            requests.append(sceneClassifier)
        }
        
        if let objectDetector = objectDetector {
            requests.append(objectDetector)
        }
        
        if let faceLandmarkDetector = faceLandmarkDetector {
            requests.append(faceLandmarkDetector)
        }
        
        // Add CoreML requests
        if let customSceneModel = customSceneModel {
            let sceneRequest = VNCoreMLRequest(model: customSceneModel)
            requests.append(sceneRequest)
        }
        
        if let compositionModel = compositionModel {
            let compositionRequest = VNCoreMLRequest(model: compositionModel)
            requests.append(compositionRequest)
        }
        
        if let aestheticsModel = aestheticsModel {
            let aestheticsRequest = VNCoreMLRequest(model: aestheticsModel)
            requests.append(aestheticsRequest)
        }
        
        try requestHandler.perform(requests)
        
        // Collect results
        for request in requests {
            observations.append(contentsOf: request.results ?? [])
        }
        
        return processVisionResults(observations, image: cgImage, timestamp: timestamp)
    }
    
    private func processVisionResults(_ observations: [VNObservation], image: CGImage, timestamp: CMTime) -> FrameAnalysis {
        var sceneClassifications: [SceneClassification] = []
        var detectedObjects: [DetectedObject] = []
        var faces: [DetectedFace] = []
        var compositionMetrics = CompositionMetrics()
        var qualityMetrics = QualityMetrics()
        
        for observation in observations {
            switch observation {
            case let classification as VNClassificationObservation:
                sceneClassifications.append(SceneClassification(
                    label: classification.identifier,
                    confidence: Double(classification.confidence),
                    category: classifySceneCategory(classification.identifier)
                ))
                
            case let rectangle as VNRectangleObservation:
                detectedObjects.append(DetectedObject(
                    boundingBox: rectangle.boundingBox,
                    label: "Rectangle",
                    confidence: Double(rectangle.confidence),
                    objectType: .geometric
                ))
                
            case let faceObservation as VNFaceObservation:
                faces.append(DetectedFace(
                    boundingBox: faceObservation.boundingBox,
                    confidence: Double(faceObservation.confidence),
                    emotion: "neutral",
                    pose: extractFacePose(faceObservation)
                ))
                
            case let coreMLObservation as VNCoreMLFeatureValueObservation:
                // Process CoreML results
                processCoreMLObservation(coreMLObservation, compositionMetrics: &compositionMetrics, qualityMetrics: &qualityMetrics)
                
            default:
                break
            }
        }
        
        // Compute additional metrics
        compositionMetrics = computeCompositionMetrics(image: image, objects: detectedObjects, faces: faces)
        qualityMetrics = computeQualityMetrics(image: image)
        
        return FrameAnalysis(
            timestamp: timestamp.seconds,
            sceneClassifications: sceneClassifications,
            detectedObjects: detectedObjects,
            faces: faces,
            compositionMetrics: compositionMetrics,
            qualityMetrics: qualityMetrics,
            colorMetrics: computeColorMetrics(image: image),
            motionMetrics: MotionMetrics(globalMotion: 0.0, localMotion: 0.0, direction: "none")
        )
    }
    
    private func aggregateFrameAnalyses(_ frameAnalyses: [FrameAnalysis], videoDuration: CMTime) -> VisualAnalysis {
        let totalFrames = frameAnalyses.count
        
        // Aggregate scene classifications
        var sceneConfidence: [String: Double] = [:]
        var totalSceneConfidence = 0.0
        
        for frame in frameAnalyses {
            for classification in frame.sceneClassifications {
                sceneConfidence[classification.label, default: 0.0] += classification.confidence
                totalSceneConfidence += classification.confidence
            }
        }
        
        let normalizedSceneConfidence = sceneConfidence.mapValues { $0 / totalSceneConfidence }
        let dominantScene = normalizedSceneConfidence.max(by: { $0.value < $1.value })?.key ?? "unknown"
        
        // Aggregate detected objects
        var objectCounts: [String: Int] = [:]
        var totalObjects = 0
        
        for frame in frameAnalyses {
            for object in frame.detectedObjects {
                objectCounts[object.label, default: 0] += 1
                totalObjects += 1
            }
        }
        
        // Aggregate composition metrics
        let avgComposition = frameAnalyses.reduce(CompositionMetrics()) { result, frame in
            CompositionMetrics(
                balance: result.balance + frame.compositionMetrics.balance,
                symmetry: result.symmetry + frame.compositionMetrics.symmetry,
                ruleOfThirds: result.ruleOfThirds + frame.compositionMetrics.ruleOfThirds,
                leadingLines: result.leadingLines + frame.compositionMetrics.leadingLines,
                framing: result.framing + frame.compositionMetrics.framing,
                depth: result.depth + frame.compositionMetrics.depth
            )
        }
        
        let finalComposition = CompositionMetrics(
            balance: avgComposition.balance / Double(totalFrames),
            symmetry: avgComposition.symmetry / Double(totalFrames),
            ruleOfThirds: avgComposition.ruleOfThirds / Double(totalFrames),
            leadingLines: avgComposition.leadingLines / Double(totalFrames),
            framing: avgComposition.framing / Double(totalFrames),
            depth: avgComposition.depth / Double(totalFrames)
        )
        
        // Aggregate quality metrics
        let avgQuality = frameAnalyses.reduce(QualityMetrics()) { result, frame in
            QualityMetrics(
                sharpness: result.sharpness + frame.qualityMetrics.sharpness,
                noise: result.noise + frame.qualityMetrics.noise,
                compression: result.compression + frame.qualityMetrics.compression,
                overallQuality: result.overallQuality + frame.qualityMetrics.overallQuality,
                technicalScore: result.technicalScore + frame.qualityMetrics.technicalScore,
                aestheticScore: result.aestheticScore + frame.qualityMetrics.aestheticScore
            )
        }
        
        let finalQuality = QualityMetrics(
            sharpness: avgQuality.sharpness / Double(totalFrames),
            noise: avgQuality.noise / Double(totalFrames),
            compression: avgQuality.compression / Double(totalFrames),
            overallQuality: avgQuality.overallQuality / Double(totalFrames),
            technicalScore: avgQuality.technicalScore / Double(totalFrames),
            aestheticScore: avgQuality.aestheticScore / Double(totalFrames)
        )
        
        return VisualAnalysis(
            sceneClassifications: normalizedSceneConfidence.map { SceneClassification(label: $0.key, confidence: $0.value, category: classifySceneCategory($0.key)) },
            dominantScene: dominantScene,
            sceneChanges: detectSceneChanges(frameAnalyses),
            averageComposition: finalComposition,
            qualityMetrics: finalQuality,
            colorPalette: extractDominantColors(frameAnalyses),
            frameAnalyses: frameAnalyses,
            visualComplexity: calculateVisualComplexity(frameAnalyses),
        )
    }
    
    // MARK: - Helper Methods
    
    private func classifySceneCategory(_ sceneLabel: String) -> SceneCategory {
        let lowercased = sceneLabel.lowercased()
        
        if lowercased.contains("indoor") || lowercased.contains("room") || lowercased.contains("office") {
            return .interior
        } else if lowercased.contains("outdoor") || lowercased.contains("sky") || lowercased.contains("landscape") {
            return .exterior
        } else if lowercased.contains("person") || lowercased.contains("face") || lowercased.contains("people") {
            return .dialogue
        } else if lowercased.contains("action") || lowercased.contains("sport") || lowercased.contains("movement") {
            return .action
        } else if lowercased.contains("nature") || lowercased.contains("animal") || lowercased.contains("plant") {
            return .nature
        } else if lowercased.contains("urban") || lowercased.contains("city") || lowercased.contains("building") {
            return .urban
        } else {
            return .other
        }
    }
    
    private func extractFaceLandmarks(_ landmarks: VNFaceLandmarks2D?) -> [CGPoint] {
        guard let landmarks = landmarks else { return [] }
        
        var points: [CGPoint] = []
        
        // Extract key landmarks
        if let leftEye = landmarks.leftEye {
            points.append(contentsOf: leftEye.normalizedPoints)
        }
        
        if let rightEye = landmarks.rightEye {
            points.append(contentsOf: rightEye.normalizedPoints)
        }
        
        if let nose = landmarks.nose {
            points.append(contentsOf: nose.normalizedPoints)
        }
        
        if let outerLips = landmarks.outerLips {
            points.append(contentsOf: outerLips.normalizedPoints)
        }
        
        return points
    }
    
    private func extractFacePose(_ faceObservation: VNFaceObservation) -> FacePose {
        return FacePose(
            yaw: faceObservation.yaw?.doubleValue ?? 0.0,
            pitch: faceObservation.pitch?.doubleValue ?? 0.0,
            roll: faceObservation.roll?.doubleValue ?? 0.0
        )
    }
    
    private func processCoreMLObservation(_ observation: VNCoreMLFeatureValueObservation, 
                                        compositionMetrics: inout CompositionMetrics,
                                        qualityMetrics: inout QualityMetrics) {
        // Process CoreML feature values to extract composition and quality metrics
        // This would be customized based on the specific models used
    }
    
    private func computeCompositionMetrics(image: CGImage, objects: [DetectedObject], faces: [DetectedFace]) -> CompositionMetrics {
        let width = Double(image.width)
        let height = Double(image.height)
        
        // Rule of thirds analysis
        let ruleOfThirds = analyzeRuleOfThirds(objects: objects, faces: faces, imageSize: CGSize(width: width, height: height))
        
        // Symmetry analysis
        let symmetry = analyzeSymmetry(objects: objects, faces: faces, imageSize: CGSize(width: width, height: height))
        
        // Balance analysis
        let balance = analyzeBalance(objects: objects, faces: faces, imageSize: CGSize(width: width, height: height))
        
        return CompositionMetrics(
            balance: balance,
            symmetry: symmetry,
            ruleOfThirds: ruleOfThirds,
            leadingLines: 0.5, // Placeholder - would require more advanced analysis
            framing: 0.6, // Placeholder
            depth: 0.7 // Placeholder
        )
    }
    
    private func computeQualityMetrics(image: CGImage) -> QualityMetrics {
        // This would implement actual image quality analysis
        // For now, returning reasonable placeholder values
        return QualityMetrics(
            sharpness: 0.8,
            noise: 0.1,
            compression: 0.05,
            overallQuality: 0.7,
            technicalScore: 0.75,
            aestheticScore: 0.65
        )
    }
    
    private func computeColorMetrics(image: CGImage) -> ColorMetrics {
        // Basic color analysis - would be enhanced with actual histogram analysis
        return ColorMetrics(
            dominantColors: ["#3498db", "#e74c3c", "#2ecc71"],
            colorTemperature: 5500,
            vibrance: 0.7,
            saturation: 0.65
        )
    }
    
    private func analyzeRuleOfThirds(objects: [DetectedObject], faces: [DetectedFace], imageSize: CGSize) -> Double {
        let thirdX = imageSize.width / 3
        let thirdY = imageSize.height / 3
        
        var score = 0.0
        var totalSubjects = 0
        
        // Check objects
        for object in objects {
            let centerX = object.boundingBox.midX * imageSize.width
            let centerY = object.boundingBox.midY * imageSize.height
            
            if abs(centerX - thirdX) < 50 || abs(centerX - 2 * thirdX) < 50 ||
               abs(centerY - thirdY) < 50 || abs(centerY - 2 * thirdY) < 50 {
                score += 1.0
            }
            totalSubjects += 1
        }
        
        // Check faces
        for face in faces {
            let centerX = face.boundingBox.midX * imageSize.width
            let centerY = face.boundingBox.midY * imageSize.height
            
            if abs(centerX - thirdX) < 50 || abs(centerX - 2 * thirdX) < 50 ||
               abs(centerY - thirdY) < 50 || abs(centerY - 2 * thirdY) < 50 {
                score += 1.0
            }
            totalSubjects += 1
        }
        
        return totalSubjects > 0 ? score / Double(totalSubjects) : 0.5
    }
    
    private func analyzeSymmetry(objects: [DetectedObject], faces: [DetectedFace], imageSize: CGSize) -> Double {
        let centerX = imageSize.width / 2
        var symmetryScore = 0.0
        var totalElements = 0
        
        // Analyze object symmetry
        for object in objects {
            let objectCenterX = object.boundingBox.midX * imageSize.width
            let distanceFromCenter = abs(objectCenterX - centerX)
            let symmetryContribution = max(0, 1.0 - (distanceFromCenter / centerX))
            symmetryScore += symmetryContribution
            totalElements += 1
        }
        
        return totalElements > 0 ? symmetryScore / Double(totalElements) : 0.5
    }
    
    private func analyzeBalance(objects: [DetectedObject], faces: [DetectedFace], imageSize: CGSize) -> Double {
        let centerX = imageSize.width / 2
        var leftWeight = 0.0
        var rightWeight = 0.0
        
        // Calculate visual weight distribution
        for object in objects {
            let objectCenterX = object.boundingBox.midX * imageSize.width
            let area = object.boundingBox.width * object.boundingBox.height * imageSize.width * imageSize.height
            let weight = sqrt(area) * object.confidence
            
            if objectCenterX < centerX {
                leftWeight += weight
            } else {
                rightWeight += weight
            }
        }
        
        let totalWeight = leftWeight + rightWeight
        if totalWeight == 0 { return 0.5 }
        
        let balance = 1.0 - abs(leftWeight - rightWeight) / totalWeight
        return balance
    }
    
    private func detectSceneChanges(_ frameAnalyses: [FrameAnalysis]) -> [SceneChange] {
        var sceneChanges: [SceneChange] = []
        
        for i in 1..<frameAnalyses.count {
            let prevFrame = frameAnalyses[i-1]
            let currentFrame = frameAnalyses[i]
            
            // Compare dominant scene classifications
            let prevDominant = prevFrame.sceneClassifications.max(by: { $0.confidence < $1.confidence })
            let currentDominant = currentFrame.sceneClassifications.max(by: { $0.confidence < $1.confidence })
            
            if let prev = prevDominant, let current = currentDominant, prev.label != current.label {
                let confidence = abs(current.confidence - prev.confidence)
                
                if confidence > 0.3 { // Threshold for scene change detection
                    sceneChanges.append(SceneChange(
                        timestamp: currentFrame.timestamp,
                        fromScene: prev.label,
                        toScene: current.label,
                        confidence: confidence,
                        changeType: .cut
                    ))
                }
            }
        }
        
        return sceneChanges
    }
    
    private func extractDominantColors(_ frameAnalyses: [FrameAnalysis]) -> [String] {
        // Placeholder implementation - would analyze actual frame colors
        return ["#1F1F1F", "#3F3F3F", "#5F5F5F", "#7F7F7F", "#9F9F9F"]
    }
    
    private func calculateVisualComplexity(_ frameAnalyses: [FrameAnalysis]) -> Double {
        let avgObjectCount = frameAnalyses.reduce(0.0) { $0 + Double($1.detectedObjects.count) } / Double(frameAnalyses.count)
        let avgFaceCount = frameAnalyses.reduce(0.0) { $0 + Double($1.faces.count) } / Double(frameAnalyses.count)
        let avgSceneCount = frameAnalyses.reduce(0.0) { $0 + Double($1.sceneClassifications.count) } / Double(frameAnalyses.count)
        
        return (avgObjectCount + avgFaceCount + avgSceneCount) / 3.0
    }
    
    private func calculateAestheticScore(_ composition: CompositionMetrics, _ quality: QualityMetrics) -> Double {
        let compositionScore = (composition.ruleOfThirds + composition.symmetry + composition.balance) / 3.0
        let qualityScore = (quality.sharpness + quality.technicalScore + quality.aestheticScore) / 3.0
        
        return (compositionScore + qualityScore) / 2.0
    }
}

// MARK: - Analysis Errors

public enum AnalysisError: Error, LocalizedError {
    case noVideoTrack
    case invalidSampleBuffer
    case imageProcessingFailed
    case modelLoadingFailed(String)
    case analysisTimeout
    
    public var errorDescription: String? {
        switch self {
        case .noVideoTrack:
            return "No video track found in the media file"
        case .invalidSampleBuffer:
            return "Invalid sample buffer for frame analysis"
        case .imageProcessingFailed:
            return "Failed to process image data"
        case .modelLoadingFailed(let modelName):
            return "Failed to load \(modelName) model"
        case .analysisTimeout:
            return "Visual analysis timed out"
        }
    }
}
