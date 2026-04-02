package main

import _ "embed"

//go:embed corpus.txt
var corpus string

// promptText returns exactly targetChars characters from the corpus,
// starting at an offset derived from epoch (using prime 7919 for distribution).
// Wraps around the corpus when targetChars > len(corpus).
func promptText(targetChars int, epoch int) string {
	n := len(corpus)
	if n == 0 || targetChars <= 0 {
		return ""
	}
	offset := (epoch * 7919) % n
	buf := make([]byte, 0, targetChars)
	pos := offset
	for len(buf) < targetChars {
		remaining := targetChars - len(buf)
		available := n - pos
		take := min(remaining, available)
		buf = append(buf, corpus[pos:pos+take]...)
		pos = (pos + take) % n
	}
	return string(buf)
}

// calibrateChars scales charCount so that the next prompt will hit targetTokens,
// given that the current charCount produced actualTokens.
// Returns charCount unchanged if actualTokens is zero.
func calibrateChars(charCount, targetTokens, actualTokens int) int {
	if actualTokens == 0 {
		return charCount
	}
	return int(float64(charCount) * float64(targetTokens) / float64(actualTokens))
}

// initialChars returns the starting character-count estimate for targetTokens.
// Uses 4 chars/token as a conservative estimate for English prose.
func initialChars(targetTokens int) int {
	return targetTokens * 4
}
