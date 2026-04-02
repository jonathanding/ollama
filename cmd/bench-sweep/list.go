package main

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

func runList(_ []string) error {
	records, err := listRuns()
	if err != nil {
		return err
	}
	printList(os.Stdout, records)
	return nil
}

func printList(w io.Writer, records []RunRecord) {
	if len(records) == 0 {
		fmt.Fprintln(w, "No benchmark runs found.")
		return
	}
	fmt.Fprintf(w, "%-24s %-28s %-12s %-20s %s\n", "NAME", "MODEL", "DATE", "SIZES", "STABLE")
	fmt.Fprintln(w, strings.Repeat("─", 95))
	for _, rec := range records {
		sizes := make([]string, len(rec.Config.Sizes))
		for i, s := range rec.Config.Sizes {
			sizes[i] = strconv.Itoa(s)
		}
		stableCount := 0
		for _, r := range rec.Results {
			if r.Stable {
				stableCount++
			}
		}
		fmt.Fprintf(w, "%-24s %-28s %-12s %-20s %d/%d\n",
			rec.Name,
			rec.Model,
			rec.Timestamp.Format("2006-01-02"),
			strings.Join(sizes, ","),
			stableCount,
			len(rec.Results),
		)
	}
}
