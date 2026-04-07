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
      optionally followed by a speaker name and/or text on the same line.
    - following lines until a blank line: transcription

    Returns a list of (start_hhmmss, end_hhmmss, speaker_name_or_None, text).
    """
    blocks = []
    i = 0
    n = len(lines)

    # Matches e.g.:
    # "00:00:09.818 --> 00:00:11.344\tMedrano"
    # "00:00:16.704 --> 00:00:19.114"
    # "00:00:43.044 --> 00:00:52.226    Americo"
    time_pattern = re.compile(
        r'^(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*'
        r'(\d{2}:\d{2}:\d{2}\.\d{3})'
        r'(?:\s+(.*))?$'  # optional trailing content (speaker name and/or text)
    )

    while i < n:
        line = lines[i].rstrip('\n')

        # Skip empty lines
        if not line.strip():
            i += 1
            continue

        m = time_pattern.match(line.strip())
        if m:
            start_raw = m.group(1)
            end_raw = m.group(2)
            trailing = m.group(3) or ""  # may contain speaker name and/or text

            start = time_to_hhmmss(start_raw)
            end = time_to_hhmmss(end_raw)

            # Try to extract a speaker name from the trailing part.
            # For now, we assume the trailing part is just the speaker name,
            # e.g. "Medrano" or "Americo". If you later have cases where
            # trailing also contains text, we can refine this.
            speaker_name = trailing.strip() if trailing.strip() else None

            # Collect following text lines until next blank line or timecode
            i += 1
            text_lines = []

            while i < n:
                curr = lines[i].rstrip('\n')
                if not curr.strip():
                    i += 1
                    break
                if time_pattern.match(curr.strip()):
                    break
                text_lines.append(curr)
                i += 1

            text = ' '.join(text_lines).strip()
            blocks.append((start, end, speaker_name, text))
        else:
            # Not a timecode line; skip
            i += 1

    return blocks

# ---------- Speaker segmentation and merging ----------

def build_speaker_segments(blocks):
    """
    From the list of (start, end, speaker_name, text) blocks, build a list of segments:
      (speaker_label, start_td, end_td, text)

    Speaker detection rules (in order of priority):

      1. If the block has a non-empty speaker_name on the timecode line
         (e.g. 'Medrano', 'Americo'), use that as the speaker label for this block
         and for subsequent blocks until it changes.

      2. If there is no speaker_name, but the text starts with a leading dash '-',
         treat this as a new unnamed speaker turn:
             - Strip the leading '-' from the text.
             - Assign a new anonymous speaker label: 'Speaker 1', 'Speaker 2', etc.

      3. If there is no speaker_name and no leading dash, treat this block as a
         continuation of the previous speaker (if any). If there is no previous
         speaker yet, label as 'Unknown'.

    The speaker_label is either:
      - the actual name from the timecode line (e.g. 'Medrano', 'Americo'), or
      - an anonymous label like 'Speaker 1', 'Speaker 2', or
      - 'Unknown' if nothing else is available.
    """
    segments = []
    last_speaker_label = None
    anonymous_speaker_count = 0  # for dash-based unnamed speakers

    for start_h, end_h, speaker_name, text in blocks:
        raw_text = text or ""
        cleaned_text = raw_text.strip()

        # Case 1: explicit speaker name on timecode line
        if speaker_name and speaker_name.strip():
            speaker_label = speaker_name.strip()
            last_speaker_label = speaker_label

        else:
            # No explicit name on this block
            # Case 2: leading dash in text indicates a new unnamed speaker
            if cleaned_text.startswith("-"):
                anonymous_speaker_count += 1
                speaker_label = f"Speaker {anonymous_speaker_count}"
                # Strip the leading '-' (first occurrence after any leading spaces)
                idx = raw_text.find("-")
                stripped = raw_text[:idx] + raw_text[idx + 1:]
                cleaned_text = stripped.strip()
                last_speaker_label = speaker_label
            else:
                # Case 3: continuation of previous speaker, or Unknown
                if last_speaker_label is not None:
                    speaker_label = last_speaker_label
                else:
                    speaker_label = "Unknown"
                    last_speaker_label = speaker_label

        start_td = hhmmss_to_timedelta(start_h)
        end_td = hhmmss_to_timedelta(end_h)

        segments.append((speaker_label, start_td, end_td, cleaned_text))

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
    for start, end, speaker_name, text in blocks:
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

st.write(
    "Optional: Merge rows based on set character targets to consolidate segments by the same speaker."
)

st.markdown(
    """
**Speaker detection (for merging mode)**

st.markdown(
    """
**Speaker detection (for merging mode)**

- The tool first looks for a **speaker name on the timecode line**, after the end time.  
  - Example:  
    `00:00:09.818 --> 00:00:11.344 ^t Medrano` -> speaker = `Medrano`  
    `00:00:43.044 --> 00:00:52.226 ^t Americo` -> speaker = `Americo`
- All blocks that share the same speaker name are treated as the **same speaker** and
  can be merged together.

- If a block does **not** have a speaker name on the timecode line, the tool then checks
  whether the block’s text **starts with a hyphen** (`-`):
  - Example text: `-When I was young I loved a tree`
  - This is treated as a **new unnamed speaker turn** (labeled `Speaker 1`, `Speaker 2`, etc.),
    and the leading `-` is removed from the text.

- If a block has **no speaker name** and its text does **not** start with `-`, it is treated
  as a **continuation of the previous speaker**. If there is no previous speaker yet, the
  speaker is labeled `"Unknown"`.

For best results, use the “Convert and merge consecutive lines by speaker” option only when
your transcript either:
- includes speaker names on the timecode lines, or
- uses a leading `-` to mark new speaker turns.
Otherwise, speaker separation may not match the actual conversation.
    """
)

mode = st.radio(
    "Conversion options",
    options=[
        "Simple conversion to TSV",
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
