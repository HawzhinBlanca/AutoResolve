import Foundation
import CoreMedia

public struct EDLExporter {
    public static func export(timeline: TimelineModel, to url: URL) throws {
        var edlContent = "TITLE: AutoResolve Timeline\n"
        edlContent += "FCM: NON-DROP FRAME\n\n"
        
        var eventNumber = 1
        
        // Export video clips
        for track in timeline.tracks {
            for clip in track.clips.sorted(by: { $0.startTime < $1.startTime }) {
                let sourceIn = formatTimecode(clip.startTime, fps: 30.0)
                let sourceOut = formatTimecode(clip.startTime + clip.duration, fps: 30.0)
                let recordIn = formatTimecode(clip.startTime, fps: 30.0)
                let recordOut = formatTimecode(clip.startTime + clip.duration, fps: 30.0)
                
                edlContent += String(format: "%03d  %@ V     C        %@ %@ %@ %@\n", 
                    eventNumber,
                    String(clip.name.prefix(8)).padding(toLength: 8, withPad: " ", startingAt: 0),
                    sourceIn, sourceOut, recordIn, recordOut
                )
                
                edlContent += "* FROM CLIP NAME: \(clip.name)\n"
                
                if let url = clip.sourceURL {
                    edlContent += "* FROM CLIP: \(url.lastPathComponent)\n"
                }
                
                edlContent += "\n"
                eventNumber += 1
            }
        }
        
        try edlContent.write(to: url, atomically: true, encoding: .utf8)
    }
    
    private static func formatTimecode(_ seconds: TimeInterval, fps: Double) -> String {
        let totalFrames = Int(seconds * fps)
        let frames = totalFrames % Int(fps)
        let totalSeconds = totalFrames / Int(fps)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secs = totalSeconds % 60
        
        return String(format: "%02d:%02d:%02d:%02d", hours, minutes, secs, frames)
    }
}