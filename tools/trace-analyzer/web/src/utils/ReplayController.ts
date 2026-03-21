import type { TimelineOp } from '../types/trace';

export interface ReplayCallbacks {
  onStep: (step: TimelineOp, index: number) => void;
  onFinish: () => void;
}

const BASE_INTERVAL = 80;

export class ReplayController {
  state: 'idle' | 'playing' | 'paused' = 'idle';
  steps: TimelineOp[];
  currentIndex = -1;
  speed = 1;

  private callbacks: ReplayCallbacks;
  private timerId: ReturnType<typeof setInterval> | null = null;

  constructor(steps: TimelineOp[], callbacks: ReplayCallbacks) {
    this.steps = steps;
    this.callbacks = callbacks;
  }

  play(): void {
    if (this.state === 'playing' || this.steps.length === 0) return;
    this.state = 'playing';
    this.currentIndex = 0;
    this.callbacks.onStep(this.steps[0], 0);
    this.startTimer();
  }

  pause(): void {
    if (this.state !== 'playing') return;
    this.state = 'paused';
    this.clearTimer();
  }

  resume(): void {
    if (this.state !== 'paused') return;
    this.state = 'playing';
    this.startTimer();
  }

  stop(): void {
    this.state = 'idle';
    this.currentIndex = -1;
    this.clearTimer();
  }

  setSpeed(n: number): void {
    this.speed = n;
    if (this.state === 'playing') {
      this.clearTimer();
      this.startTimer();
    }
  }

  seekTo(index: number): void {
    if (index < 0 || index >= this.steps.length) return;
    this.currentIndex = index;
    this.callbacks.onStep(this.steps[index], index);
  }

  dispose(): void {
    this.clearTimer();
  }

  private startTimer(): void {
    this.clearTimer();
    this.timerId = setInterval(() => this.advance(), BASE_INTERVAL / this.speed);
  }

  private clearTimer(): void {
    if (this.timerId !== null) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
  }

  private advance(): void {
    const next = this.currentIndex + 1;
    if (next >= this.steps.length) {
      this.clearTimer();
      this.state = 'idle';
      this.callbacks.onFinish();
      return;
    }
    this.currentIndex = next;
    this.callbacks.onStep(this.steps[next], next);
  }
}
