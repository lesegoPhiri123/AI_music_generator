def align_lines_to_beats(lines, beats):
    return list(zip(beats[:len(lines)], lines))
