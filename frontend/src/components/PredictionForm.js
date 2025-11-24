/**
 * PredictionForm Component
 * Form for inputting repository information and triggering fault localization
 */

import React, { useState } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

function PredictionForm({ onPredictionComplete, isLoading, setIsLoading }) {
  const [formData, setFormData] = useState({
    repository_path: '',
    failing_test: '',
    error_message: '',
    stack_trace: '',
    max_candidates: 50,
  });

  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        repository_path: formData.repository_path,
        failing_test: formData.failing_test,
        error_message: formData.error_message || null,
        stack_trace: formData.stack_trace || null,
        max_candidates: parseInt(formData.max_candidates),
      });

      onPredictionComplete(response.data);
    } catch (err) {
      console.error('Prediction error:', err);
      setError(
        err.response?.data?.detail ||
          'Failed to analyze repository. Please check your inputs and try again.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadExample = () => {
    setFormData({
      repository_path: 'C:\\Users\\gupta\\OneDrive\\Desktop\\sdl_final\\backend',
      failing_test: 'test_user_authentication',
      error_message: 'AssertionError: Expected True but got False',
      stack_trace: 'File "main.py", line 42, in predict_faults\n    assert results is not None',
      max_candidates: 50,
    });
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">
          Repository Analysis
        </h2>
        <button
          type="button"
          onClick={handleLoadExample}
          className="text-sm text-blue-600 hover:text-blue-800 font-medium"
        >
          Load Example
        </button>
      </div>

      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-start">
            <svg
              className="w-5 h-5 text-red-600 mt-0.5 mr-3"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            <div>
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Repository Path */}
        <div>
          <label
            htmlFor="repository_path"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Repository Path <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="repository_path"
            name="repository_path"
            value={formData.repository_path}
            onChange={handleChange}
            required
            placeholder="e.g., C:\Users\user\project or /home/user/project"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
          />
          <p className="mt-1 text-xs text-gray-500">
            Absolute path to your repository
          </p>
        </div>

        {/* Failing Test */}
        <div>
          <label
            htmlFor="failing_test"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Failing Test Name <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="failing_test"
            name="failing_test"
            value={formData.failing_test}
            onChange={handleChange}
            required
            placeholder="e.g., test_user_login"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
          />
          <p className="mt-1 text-xs text-gray-500">
            Name of the test that failed
          </p>
        </div>

        {/* Error Message */}
        <div>
          <label
            htmlFor="error_message"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Error Message <span className="text-gray-400">(Optional)</span>
          </label>
          <input
            type="text"
            id="error_message"
            name="error_message"
            value={formData.error_message}
            onChange={handleChange}
            placeholder="e.g., AssertionError: Expected 200, got 404"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
          />
        </div>

        {/* Stack Trace */}
        <div>
          <label
            htmlFor="stack_trace"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Stack Trace <span className="text-gray-400">(Optional)</span>
          </label>
          <textarea
            id="stack_trace"
            name="stack_trace"
            value={formData.stack_trace}
            onChange={handleChange}
            rows="4"
            placeholder='File "auth.py", line 42, in validate_user&#10;    assert user.is_active&#10;AssertionError'
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition font-mono text-sm"
          />
          <p className="mt-1 text-xs text-gray-500">
            Paste the full stack trace for better accuracy
          </p>
        </div>

        {/* Max Candidates */}
        <div>
          <label
            htmlFor="max_candidates"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Max Candidates: {formData.max_candidates}
          </label>
          <input
            type="range"
            id="max_candidates"
            name="max_candidates"
            value={formData.max_candidates}
            onChange={handleChange}
            min="10"
            max="200"
            step="10"
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>10</span>
            <span>200</span>
          </div>
        </div>

        {/* Submit Button */}
        <div className="pt-4">
          <button
            type="submit"
            disabled={isLoading}
            className={`w-full py-4 px-6 rounded-lg font-semibold text-white transition-all ${
              isLoading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 shadow-lg hover:shadow-xl'
            }`}
          >
            {isLoading ? (
              <span className="flex items-center justify-center">
                <svg
                  className="animate-spin h-5 w-5 mr-3"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                    fill="none"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                Analyzing Repository...
              </span>
            ) : (
              <span className="flex items-center justify-center">
                <svg
                  className="w-5 h-5 mr-2"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"
                  />
                </svg>
                Analyze Repository
              </span>
            )}
          </button>
        </div>
      </form>

      {/* Info Box */}
      <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start">
          <svg
            className="w-5 h-5 text-blue-600 mt-0.5 mr-3"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
              clipRule="evenodd"
            />
          </svg>
          <div className="text-sm text-blue-800">
            <p className="font-medium">How it works:</p>
            <ul className="mt-2 space-y-1 list-disc list-inside text-blue-700">
              <li>Extracts candidate lines from your repository</li>
              <li>Generates CodeBERT embeddings for semantic analysis</li>
              <li>Uses ML model to predict fault locations</li>
              <li>Returns ranked list of suspicious lines</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default PredictionForm;
