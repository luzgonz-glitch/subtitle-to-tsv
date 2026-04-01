import re
import io
import zipfile
from datetime import timedelta

import streamlit as st


# ---------- Time helpers ----------

def time_to_hhmmss(time_str: str) -> str:
    """
    Convert 'HH:MM:SS.mmm' to 'HH:MM:SS' (truncate milliseconds).
    """
    base = time_str.split('.')[0]
    return base

def hhmmss_to_timedelta(hhmmss: str) -> timedelta:
    """
    Convert 'HH:MM:SS' to a datetime.timedelta.
    """
    h, m, s = hhmmss.split(":")
    return timedelta(hours=int(h), minutes=int(m), seconds=int(s))

def timedelta_to_hhmmss(td: timedelta) -> str:
    """
    Convert a datetime.timedelta to 'HH:MM:SS'.
    """
    total_seconds = int(td.total_seconds())
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------- Parsing original cues ----------

def parse_blocks(lines):
    """
    Parse the text into blocks:
    - 1st line: timecode line 'HH:MM:SS.mmm --> HH:MM:SS.mmm'
    - following lines until a blank line: transcription
    Returns a list of (start_hhmmss, end_hhmmss, text).
    """
    blocks = []
    i = 0
    n = len(lines)

    time_pattern = re.compile(
        r'^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}'
    )

    while i < n:
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # If the line is a timecode line
        if time_pattern.match(line):
            time_line = line
            # Collect following text lines until next blank line or timecode
            i += 1
            text_lines = []
            while i < n:
                curr = lines[i]
                # Stop at blank line
                if curr.strip() == '':
                    i += 1
                    break
                # Stop if the next line looks like a timecode (start of next block)
                if time_pattern.match(curr.strip()):
                    break
                text_lines.append(curr.rstrip('\n'))
                i += 1

            # Parse the timecodes
            start_raw, end_raw = [part.strip() for part in time_line.split('-->')]
            start = time_to_hhmmss(start_raw)
            end = time_to_hhmmss(end_raw)
            text = ' '.join(text_lines).strip()

            blocks.append((start, end, text))
        else:
            # If the line is not empty and not a timecode, skip it
            i += 1

    return blocks


# ---------- Speaker segmentation and merging ----------

def build_speaker_segments(blocks):
    """
    From the list of (start, end, text) blocks, build a list of segments:
      (speaker_id, start_td, end_td, text)
    using leading '-' at the start of the text as new speaker markers.
    Speakers are labeled S1, S2, S3,... in order of appearance.
    """
    segments = []
    current_speaker_index = 0  # 0 means "no speaker yet"

    for start_h, end_h, text in blocks:
        # Determine if this block starts a new speaker turn
        is_new_speaker = text.startswith("-")
        cleaned_text = text.lstrip("-").strip()

        if is_new_speaker or current_speaker_index == 0:
            # New speaker
            current_speaker_index += 1
        speaker_id = f"S{current_speaker_index}"

        start_td = hhmmss_to_timedelta(start_h)
        end_td = hhmmss_to_timedelta(end_h)

        segments.append((speaker_id, start_td, end_td, cleaned_text))

    return segments


def merge_segments_by_speaker(
    segments,
    target_chars: int = 250,
    max_chars: int = 300,
):
    """
    Merge consecutive segments from the same speaker into larger blocks
    of approximately target_chars characters (up to max_chars).
    segments: list of (speaker_id, start_td, end_td, text)
    Returns list of merged blocks: (speaker_id, start_td, end_td, merged_text)
    """
    if not segments:
        return []

    merged = []
    current_speaker = None
    current_start = None
    current_end = None
    current_text_parts = []

    def flush_current():
        nonlocal current_speaker, current_start, current_end, current_text_parts
        if current_speaker is not None:
            merged_text = " ".join(current_text_parts).strip()
            merged.append((current_speaker, current_start, current_end, merged_text))
        current_speaker = None
        current_start = None
        current_end = None
        current_text_parts = []

    for speaker_id, start_td, end_td, text in segments:
        if current_speaker is None:
            # Start a new merged block
            current_speaker = speaker_id
            current_start = start_td
            current_end = end_td
            current_text_parts = [text]
            continue

        # If speaker changes, flush and start new
        if speaker_id != current_speaker:
            flush_current()
            current_speaker = speaker_id
            current_start = start_td
            current_end = end_td
            current_text_parts = [text]
            continue

        # Same speaker: decide whether to merge or flush based on length
        current_text = " ".join(current_text_parts)
        new_length = len(current_text) + 1 + len(text) if current_text else len(text)

        if new_length <= max_chars:
            # Merge into current block
            current_text_parts.append(text)
            current_end = end_td
        else:
            # If current block is already reasonably long, flush and start new
            if len(current_text) >= target_chars:
                flush_current()
                current_speaker = speaker_id
                current_start = start_td
                current_end = end_td
                current_text_parts = [text]
            else:
                # If current block is still short but adding this would exceed max_chars,
                # we still flush to avoid overly long blocks.
                flush_current()
                current_speaker = speaker_id
                current_start = start_td
                current_end = end_td
                current_text_parts = [text]

    # Flush the last block
    flush_current()
    return merged


# ---------- Conversion to TSV (two modes) ----------

def convert_to_tsv_simple(file_content: str) -> str:
    """
    Original behavior: one row per cue/block, no merging, no speaker column.
    """
    lines = file_content.splitlines(keepends=True)
    blocks = parse_blocks(lines)

    output = io.StringIO()
    output.write(
        "Start Timestamp (HH:MM:SS)\t"
        "Stop Timestamp (HH:MM:SS)\t"
        "Transcription of the audio byte\n"
    )
    for start, end, text in blocks:
        safe_text = text.replace('\t', ' ').replace('\n', ' ')
        output.write(f"{start}\t{end}\t{safe_text}\n")

    return output.getvalue()


def convert_to_tsv_merged(
    file_content: str,
    target_chars: int = 250,
    max_chars: int = 300,
) -> str:
    """
    Merged behavior: merge consecutive lines from the same speaker into
    ~target_chars character blocks (never exceeding max_chars).
    Adds a 'Speaker' column.
    """
    lines = file_content.splitlines(keepends=True)
    blocks = parse_blocks(lines)

    # Build speaker segments from blocks
    segments = build_speaker_segments(blocks)

    # Merge segments by speaker into larger blocks
    merged_segments = merge_segments_by_speaker(
        segments,
        target_chars=target_chars,
        max_chars=max_chars,
    )

    # Build TSV in memory
    output = io.StringIO()
    output.write(
        "Start Timestamp (HH:MM:SS)\t"
        "Stop Timestamp (HH:MM:SS)\t"
        "Speaker\t"
        "Transcription of the audio byte\n"
    )
    for speaker_id, start_td, end_td, text in merged_segments:
        safe_text = text.replace('\t', ' ').replace('\n', ' ')
        start_str = timedelta_to_hhmmss(start_td)
        end_str = timedelta_to_hhmmss(end_td)
        output.write(f"{start_str}\t{end_str}\t{speaker_id}\t{safe_text}\n")

    return output.getvalue()


# ---------- Streamlit UI ----------

st.title("Transcript to TSV Converter")

st.write(
    "Upload one or more VTT or TXT transcript files with `HH:MM:SS.mmm` timecodes and convert them to TSV with `HH:MM:SS` timcodes."
)
st.markdown(
    """
**Speaker detection (for merging mode)**

- The tool treats any block whose text **starts with a hyphen** (`-`) as a **new speaker turn**.
  - Example: `-When I was young I loved a tree`
- Blocks **without** a leading hyphen are treated as a **continuation of the same speaker**.

If your transcript does not use a leading `-` to mark new speakers, the “Convert and merge consecutive lines by speaker” option will still work,
but all lines will be treated as coming from a single speaker and only merged by character length.
    """
)

mode = st.radio(
    "Merging options",
    options=[
        "Simple conversion",
        "Convert and merge consecutive lines by speaker (character-based blocks)",
    ]
)

if mode == "Convert and merge consecutive lines by speaker (character-based blocks)":
    target_chars = st.number_input(
        "Target characters per transcription block",
        min_value=50,
        max_value=2000,
        value=250,
        step=10,
        help="The converter will try to keep merged blocks around this length."
    )
    max_chars = st.number_input(
        "Maximum characters per transcription block",
        min_value=target_chars,
        max_value=4000,
        value=int(target_chars * 1.2),
        step=10,
        help="Blocks will never exceed this length. Must be >= target characters."
    )
else:
    target_chars = None
    max_chars = None

uploaded_files = st.file_uploader(
    "Choose one or more.vtt or.txt files",
    type=["vtt", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    tsv_results = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        try:
            file_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            st.error(f"Could not decode file {uploaded_file.name} as UTF-8.")
            continue

        if mode == "Simple conversion":
            tsv_text = convert_to_tsv_simple(file_text)
        else:
            tsv_text = convert_to_tsv_merged(
                file_text,
                target_chars=int(target_chars),
                max_chars=int(max_chars),
            )

        tsv_results.append((uploaded_file.name, tsv_text))

    # Show preview of the first file
    first_name, first_tsv = tsv_results[0]
    st.subheader(f"Preview of TSV output for {first_name} (first 20 lines)")
    preview_lines = "\n".join(first_tsv.splitlines()[:20])
    st.text(preview_lines)

    # If only one file, offer direct TSV download
    if len(tsv_results) == 1:
        base_name = first_name.rsplit(".", 1)[0] + ".tsv"
        st.download_button(
            label=f"Download TSV for {first_name}",
            data=first_tsv.encode("utf-8"),
            file_name=base_name,
            mime="text/tab-separated-values"
        )
    else:
        # Multiple files: create a ZIP in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for original_name, tsv_text in tsv_results:
                tsv_name = original_name.rsplit(".", 1)[0] + ".tsv"
                zipf.writestr(tsv_name, tsv_text)

        zip_buffer.seek(0)
        st.download_button(
            label="Download all TSV files as ZIP",
            data=zip_buffer,
            file_name="converted_tsv_files.zip",
            mime="application/zip"
        )
