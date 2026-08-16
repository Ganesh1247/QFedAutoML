import React, { Component, ErrorInfo, ReactNode } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

interface Props {
  children: ReactNode
  fallbackName?: string
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught error:', error, errorInfo)
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null })
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="glass-panel p-8 max-w-lg mx-auto my-12 text-center space-y-4 border-rose-500/40 animate-fadeIn">
          <div className="w-12 h-12 rounded-full bg-rose-500/10 border border-rose-500/30 flex items-center justify-center mx-auto text-rose-400">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-display font-bold text-white">
            {this.props.fallbackName || 'Component'} Render Alert
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed font-mono">
            {this.state.error?.message || 'An unexpected rendering error occurred in this view.'}
          </p>
          <button
            onClick={this.handleReset}
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-cyan-400 font-semibold text-xs border border-slate-700 inline-flex items-center gap-2 transition-all cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Reset View</span>
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
