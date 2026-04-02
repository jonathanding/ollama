package main

import (
	"strings"
	"testing"
	"time"
)

func TestPrintList_Empty(t *testing.T) {
	var sb strings.Builder
	printList(&sb, []RunRecord{})
	if !strings.Contains(sb.String(), "No benchmark runs found") {
		t.Errorf("expected empty message, got: %s", sb.String())
	}
}

func TestPrintList_ShowsRuns(t *testing.T) {
	records := []RunRecord{
		{
			Name:      "baseline",
			Model:     "qwen3",
			Timestamp: time.Date(2026, 4, 2, 10, 0, 0, 0, time.UTC),
			Config:    RunConfig{Sizes: []int{512, 1024, 2048}},
			Results: []SizeResult{
				{Stable: true},
				{Stable: true},
				{Stable: false},
			},
		},
	}
	var sb strings.Builder
	printList(&sb, records)
	out := sb.String()

	for _, want := range []string{"baseline", "qwen3", "2026-04-02", "512,1024,2048", "2/3"} {
		if !strings.Contains(out, want) {
			t.Errorf("expected %q in list output:\n%s", want, out)
		}
	}
}

func TestPrintList_ColumnHeaders(t *testing.T) {
	records := []RunRecord{{
		Name: "r", Model: "m", Timestamp: time.Now(),
		Config: RunConfig{Sizes: []int{512}},
	}}
	var sb strings.Builder
	printList(&sb, records)
	out := sb.String()
	for _, col := range []string{"NAME", "MODEL", "DATE", "SIZES", "STABLE"} {
		if !strings.Contains(out, col) {
			t.Errorf("expected column header %q:\n%s", col, out)
		}
	}
}
