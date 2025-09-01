import SwiftUI
import AutoResolveCore

struct MediaPoolView: View {
    @EnvironmentObject var appState: AppState
    @State private var searchText = ""
    
    var body: some View {
        VStack {
            SearchBar(text: $searchText)
            
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 100))]) {
                    ForEach(filteredMedia, id: \.id) { item in
                        MediaThumbnail(item: item)
                    }
                }
            }
        }
        .padding()
    }
    
    private var filteredMedia: [MediaItem] {
        guard let pool = appState.currentProject?.mediaPool else { return [] }
        
        if searchText.isEmpty {
            return pool.items
        }
        
        return pool.items.filter { item in
            item.name.localizedCaseInsensitiveContains(searchText)
        }
    }
}

struct MediaThumbnail: View {
    let item: MediaItem
    
    var body: some View {
        VStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.gray.opacity(0.3))
                .frame(width: 100, height: 75)
                .overlay(
                    Image(systemName: iconName)
                        .font(.largeTitle)
                        .foregroundColor(.gray)
                )
            
            Text(item.name)
                .font(.caption)
                .lineLimit(1)
        }
        .draggable(item.url)
    }
    
    private var iconName: String {
        switch item.type {
        case .video: return "video"
        case .audio: return "waveform"
        case .image: return "photo"
        default: return "doc"
        }
    }
}

struct SearchBar: View {
    @Binding var text: String
    
    var body: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundColor(.gray)
            
            TextField("Search media...", text: $text)
                .textFieldStyle(RoundedBorderTextFieldStyle())
        }
    }
}
