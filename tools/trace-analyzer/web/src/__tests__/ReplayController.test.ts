import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ReplayController } from '../utils/ReplayController';
import type { TimelineOp } from '../types/trace';

function makeOps(n: number, pass = 0): TimelineOp[] {
  return Array.from({ length: n }, (_, i) => ({
    pass,
    seq: i,
    name: `op_${i}`,
    op: 'MUL_MAT',
    backend: 'CUDA0',
    t_start: i * 1000,
    t_end: (i + 1) * 1000,
  }));
}

describe('ReplayController', () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });

  it('starts in idle state', () => {
    const ctrl = new ReplayController(makeOps(5), { onStep: vi.fn(), onFinish: vi.fn() });
    expect(ctrl.state).toBe('idle');
    expect(ctrl.currentIndex).toBe(-1);
  });

  it('play transitions to playing and calls onStep', () => {
    const onStep = vi.fn();
    const onFinish = vi.fn();
    const ops = makeOps(3);
    const ctrl = new ReplayController(ops, { onStep, onFinish });

    ctrl.play();
    expect(ctrl.state).toBe('playing');
    expect(onStep).toHaveBeenCalledWith(ops[0], 0);

    vi.advanceTimersByTime(80);
    expect(onStep).toHaveBeenCalledWith(ops[1], 1);

    vi.advanceTimersByTime(80);
    expect(onStep).toHaveBeenCalledWith(ops[2], 2);

    vi.advanceTimersByTime(80);
    expect(onFinish).toHaveBeenCalled();
    expect(ctrl.state).toBe('idle');
  });

  it('pause and resume work', () => {
    const onStep = vi.fn();
    const ctrl = new ReplayController(makeOps(10), { onStep, onFinish: vi.fn() });

    ctrl.play();
    vi.advanceTimersByTime(80);
    ctrl.pause();
    expect(ctrl.state).toBe('paused');

    const callCount = onStep.mock.calls.length;
    vi.advanceTimersByTime(400);
    expect(onStep.mock.calls.length).toBe(callCount);

    ctrl.resume();
    expect(ctrl.state).toBe('playing');
    vi.advanceTimersByTime(80);
    expect(onStep.mock.calls.length).toBe(callCount + 1);
  });

  it('stop resets to idle', () => {
    const onStep = vi.fn();
    const ctrl = new ReplayController(makeOps(10), { onStep, onFinish: vi.fn() });
    ctrl.play();
    vi.advanceTimersByTime(160);
    ctrl.stop();
    expect(ctrl.state).toBe('idle');
    expect(ctrl.currentIndex).toBe(-1);
  });

  it('setSpeed changes interval', () => {
    const onStep = vi.fn();
    const ctrl = new ReplayController(makeOps(20), { onStep, onFinish: vi.fn() });
    ctrl.setSpeed(2);
    ctrl.play();
    vi.advanceTimersByTime(40);
    expect(onStep).toHaveBeenCalledTimes(2);
  });

  it('seekTo updates index immediately', () => {
    const onStep = vi.fn();
    const ctrl = new ReplayController(makeOps(10), { onStep, onFinish: vi.fn() });
    ctrl.play();
    ctrl.pause();
    onStep.mockClear();

    ctrl.seekTo(5);
    expect(ctrl.currentIndex).toBe(5);
    expect(onStep).toHaveBeenCalledWith(expect.objectContaining({ seq: 5 }), 5);
  });
});
