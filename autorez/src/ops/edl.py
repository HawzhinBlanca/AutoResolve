import logging

logger = logging.getLogger(__name__)

#!/usr/bin/env python3
"""
EDL (Edit Decision List) generator for Resolve import
Supports both EDL and FCXML formats
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET
from xml.dom import minidom

def timecode_from_seconds(seconds: float, fps: int = 30) -> str:
    """Convert seconds to timecode format HH:MM:SS:FF"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    frames = int((seconds % 1) * fps)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"

def generate_edl(
    cuts: List[Dict],
    fps: int = 30,
    video_path: str = None,
    output_path: str = None,
    transitions: List[Dict] = None
) -> str:
    """
    Generate EDL from cuts data
    
    Args:
        cuts: List of cut dictionaries with keep regions
        fps: Frame rate
        video_path: Optional path to source video
        output_path: Optional output EDL path
        transitions: Optional list of transitions
        
    Returns:
        EDL content as string
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # EDL header
    lines = [
        "TITLE: AutoResolve Edit",
        "",
        "FCM: NON-DROP FRAME",
        ""
    ]
    
    # Get video name
    video_name = Path(video_path).name
    
    # Generate edit events
    event_num = 1
    timeline_pos = 0.0
    
    for i, region in enumerate(cuts.get("keep", [])):
        t0 = region["t0"]
        t1 = region["t1"]
        duration = t1 - t0
        
        # Source timecodes
        src_in = timecode_from_seconds(t0, fps)
        src_out = timecode_from_seconds(t1, fps)
        
        # Timeline timecodes
        tl_in = timecode_from_seconds(timeline_pos, fps)
        tl_out = timecode_from_seconds(timeline_pos + duration, fps)
        
        # Check for transition
        transition_info = "C"
        transition_duration = 0
        if transitions:
            for t in transitions:
                if t.get("clip_index") == i:
                    if t.get("type") == "dissolve":
                        transition_info = "D"
                        transition_duration = int(t.get("duration_frames", 0))
                        break

        # EDL event line
        line = f"{event_num:03d}  {video_name[:8]:8s} V     {transition_info}        {src_in} {src_out} {tl_in} {tl_out}"
        if transition_info == "D":
            line = f"{event_num:03d}  {video_name[:8]:8s} V     {transition_info}  {transition_duration:03d}  {src_in} {src_out} {tl_in} {tl_out}"

        lines.append(line)
        
        # Add source file reference
        lines.append(f"* FROM CLIP NAME: {video_name}")
        lines.append("")
        
        event_num += 1
        timeline_pos += duration
    
    # Write EDL
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    
    logger.info(f"Generated EDL with {event_num - 1} events: {output_path}")
    
    return {
        "success": True,
        "path": output_path,
        "events": event_num - 1,
        "format": "edl"
    }

def generate_fcxml(
    video_path: str,
    cuts: Dict,
    output_path: str,
    fps: int = 30
) -> Dict:
    """
    Generate FCXML (Final Cut Pro XML) from cuts data
    
    Args:
        video_path: Path to source video
        cuts: Cuts dictionary with keep regions
        output_path: Output FCXML path
        fps: Frame rate
        
    Returns:
        Result dictionary
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create FCXML structure
    fcpxml = ET.Element('fcpxml', version="1.9")
    
    # Add resources
    resources = ET.SubElement(fcpxml, 'resources')
    
    # Add format
    format_elem = ET.SubElement(resources, 'format', {
        'id': 'r1',
        'name': f'FFVideoFormat{fps}p',
        'frameDuration': f'1/{fps}s',
        'width': '1920',
        'height': '1080'
    })
    
    # Add asset
    asset = ET.SubElement(resources, 'asset', {
        'id': 'r2',
        'name': Path(video_path).name,
        'src': f'file://{Path(video_path).absolute()}',
        'hasVideo': '1',
        'hasAudio': '1',
        'format': 'r1'
    })
    
    # Add library
    library = ET.SubElement(fcpxml, 'library')
    
    # Add event
    event = ET.SubElement(library, 'event', name='AutoResolve Import')
    
    # Add project
    project = ET.SubElement(event, 'project', name='AutoResolve Edit')
    
    # Add sequence
    sequence = ET.SubElement(project, 'sequence', format='r1')
    
    # Add spine
    spine = ET.SubElement(sequence, 'spine')
    
    # Add clips for each keep region
    timeline_pos = 0
    for i, region in enumerate(cuts.get("keep", [])):
        t0 = region["t0"]
        t1 = region["t1"]
        duration = t1 - t0
        
        # Convert to frame units
        start_frames = int(t0 * fps)
        duration_frames = int(duration * fps)
        
        # Add clip
        clip = ET.SubElement(spine, 'clip', {
            'name': f'Clip {i+1}',
            'offset': f'{int(timeline_pos * fps)}/{fps}s',
            'duration': f'{duration_frames}/{fps}s',
            'start': f'{start_frames}/{fps}s'
        })
        
        # Add asset reference
        asset_clip = ET.SubElement(clip, 'asset-clip', {
            'ref': 'r2',
            'offset': f'{start_frames}/{fps}s',
            'duration': f'{duration_frames}/{fps}s'
        })
        
        timeline_pos += duration
    
    # Pretty print XML
    xml_str = ET.tostring(fcpxml, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")
    
    # Write FCXML
    with open(output_path, 'w') as f:
        f.write(pretty_xml)
    
    logger.info(f"Generated FCXML with {len(cuts.get('keep', []))} clips: {output_path}")
    
    return {
        "success": True,
        "path": output_path,
        "clips": len(cuts.get("keep", [])),
        "format": "fcxml"
    }

def generate_from_shorts(
    shorts_index: Dict,
    output_path: str,
    format: str = "edl",
    fps: int = 30
) -> Dict:
    """
    Generate EDL/FCXML from shorts index
    
    Args:
        shorts_index: Shorts index dictionary
        output_path: Output path
        format: 'edl' or 'fcxml'
        fps: Frame rate
        
    Returns:
        Result dictionary
    """
    # Convert shorts to cuts format
    cuts = {
        "keep": [
            {
                "t0": short["start"],
                "t1": short["end"]
            }
            for short in shorts_index.get("shorts", [])
        ]
    }
    
    video_path = shorts_index.get("source", "unknown.mp4")
    
    if format == "fcxml":
        return generate_fcxml(video_path, cuts, output_path, fps)
    else:
        return generate_edl(video_path, cuts, output_path, fps)


def main():
    """CLI entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="Generate EDL/FCXML")
    parser.add_argument("--cuts", help="Cuts JSON path")
    parser.add_argument("--shorts", help="Shorts index JSON path")
    parser.add_argument("--video", help="Source video path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--format", choices=['edl', 'fcxml'], default='edl', help="Output format")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    
    args = parser.parse_args()
    
    if args.shorts:
        # Generate from shorts
        with open(args.shorts) as f:
            shorts_index = json.load(f)
        result = generate_from_shorts(shorts_index, args.output, args.format, args.fps)
    elif args.cuts:
        # Generate from cuts
        with open(args.cuts) as f:
            cuts = json.load(f)
        video_path = args.video or "source.mp4"
        
        if args.format == "fcxml":
            result = generate_fcxml(video_path, cuts, args.output, args.fps)
        else:
            result = generate_edl(video_path, cuts, args.output, args.fps)
    else:
        logger.error("Error: Either --cuts or --shorts required")
        return 1
    
    if result["success"]:
        logger.info(f"✓ Generated {result['format'].upper()}")
        return 0
    else:
        return 1


def parse_edl(edl_path: str) -> Dict:
    """Parse EDL file back to edit data"""
    clips = []
    timeline_duration = 0.0
    frame_rate = 29.97
    
    try:
        with open(edl_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.startswith('001'):  # Edit entry
                parts = line.split()
                if len(parts) >= 8:
                    clips.append({
                        "name": parts[2] if len(parts) > 2 else f"clip_{len(clips)+1}",
                        "source_in": float(parts[4]) if len(parts) > 4 else 0.0,
                        "source_out": float(parts[5]) if len(parts) > 5 else 10.0,
                        "timeline_in": float(parts[6]) if len(parts) > 6 else 0.0,
                        "timeline_out": float(parts[7]) if len(parts) > 7 else 10.0,
                        "track": "V1"
                    })
        
        if clips:
            timeline_duration = max(clip["timeline_out"] for clip in clips)
            
    except Exception as e:
        logger.error(f"Error parsing EDL: {e}")
    
    return {
        "clips": clips,
        "timeline_duration": timeline_duration,
        "frame_rate": frame_rate
    }

def generate_fcpxml(project_data: Dict) -> str:
    """Generate Final Cut Pro XML"""
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.10">
    <resources>
        <format id="r1" name="FFVideoFormat1080p2997" frameDuration="1001/30000s" width="1920" height="1080"/>
    </resources>
    <library>
        <event name="{project_data['name']}">
            <project name="{project_data['name']}">
                <sequence format="r1" duration="{project_data['timeline']['duration']}">
                    <spine>
                        {''.join([
                            f'<video ref="r{i+2}" offset="{clip["start"]}" duration="{clip["duration"]}"/>'
                            for i, clip in enumerate(project_data['timeline']['clips'])
                        ])}
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>'''


if __name__ == "__main__":
    sys.exit(main())