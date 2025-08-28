import SwiftUI

public extension Color {
    static var davinciHighlight: Color {
        Color(red: 0.95, green: 0.55, blue: 0.12)
    }
    static var davinciPanel: Color { Color(red: 0.12, green: 0.12, blue: 0.12) }
    static var davinciBackground: Color { Color(red: 0.10, green: 0.10, blue: 0.10) }
    static var davinciDark: Color { Color(red: 0.08, green: 0.08, blue: 0.08) }
    static var davinciText: Color { Color.white.opacity(0.9) }
    static var davinciTextSecondary: Color { Color.white.opacity(0.6) }
}

public extension Animation {
    static var davinciQuick: Animation { .easeInOut(duration: 0.12) }
}


