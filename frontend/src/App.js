/**
 * Main React Application Component
 * AI Fault Localization Tool
 */

import React, { useState } from 'react';
import PredictionForm from './components/PredictionForm';
import ResultsDisplay from './components/ResultsDisplay';
import HistoryViewer from './components/HistoryViewer';

function App() {
  const [activeTab, setActiveTab] = useState('predict'); // 'predict' or 'history'
  const [predictionResults, setPredictionResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredictionComplete = (results) => {
    setPredictionResults(results);
  };

  const handleReset = () => {
    setPredictionResults(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="bg-gradient-to-r from-blue-500 to-indigo-600 p-3 rounded-lg">
                <svg
                  className="w-8 h-8 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
                  />
                </svg>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">
                  AI Fault Localization
                </h1>
                <p className="text-sm text-gray-600 mt-1">
                  Powered by CodeBERT & Machine Learning
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                Online
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        <div className="bg-white rounded-lg shadow-sm p-1 inline-flex space-x-1">
          <button
            onClick={() => setActiveTab('predict')}
            className={`px-6 py-2 rounded-md font-medium transition-all ${
              activeTab === 'predict'
                ? 'bg-blue-500 text-white shadow-md'
                : 'text-gray-700 hover:bg-gray-100'
            }`}
          >
            <span className="flex items-center space-x-2">
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                />
              </svg>
              <span>New Analysis</span>
            </span>
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`px-6 py-2 rounded-md font-medium transition-all ${
              activeTab === 'history'
                ? 'bg-blue-500 text-white shadow-md'
                : 'text-gray-700 hover:bg-gray-100'
            }`}
          >
            <span className="flex items-center space-x-2">
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span>History</span>
            </span>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'predict' ? (
          <div className="space-y-6">
            {!predictionResults ? (
              <PredictionForm
                onPredictionComplete={handlePredictionComplete}
                isLoading={isLoading}
                setIsLoading={setIsLoading}
              />
            ) : (
              <ResultsDisplay
                results={predictionResults}
                onReset={handleReset}
              />
            )}
          </div>
        ) : (
          <HistoryViewer onSelectPrediction={handlePredictionComplete} />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-600">
            <p>
              AI Fault Localization Tool &copy; 2025 | Built with FastAPI,
              React & CodeBERT
            </p>
            <p className="mt-2 text-xs text-gray-500">
              Using microsoft/codebert-base for semantic code analysis
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
