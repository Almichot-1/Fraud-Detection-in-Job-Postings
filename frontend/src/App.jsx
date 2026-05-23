import React, { useState } from 'react';
import './App.css';

function App() {
  // Form State
  const [formData, setFormData] = useState({
    title: '',
    company_profile: '',
    description: '',
    requirements: '',
    benefits: '',
    employment_type: 'Full-time',
    required_experience: 'Entry level',
    has_company_logo: false,
    has_questions: false,
    telecommuting: false
  });

  // UI Flow States
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // Handle Input Changes
  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  // Submit Data to Flask API Backend
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.title.strip && !formData.title.trim()) {
      setError("Please fill out at least the Job Title field.");
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          title: formData.title,
          company_profile: formData.company_profile,
          description: formData.description,
          requirements: formData.requirements,
          benefits: formData.benefits,
          employment_type: formData.employment_type,
          required_experience: formData.required_experience,
          has_company_logo: formData.has_company_logo ? 1 : 0,
          has_questions: formData.has_questions ? 1 : 0,
          telecommuting: formData.telecommuting ? 1 : 0
        })
      });

      const data = await response.json();
      if (response.ok && data.status === 'success') {
        setResult(data);
      } else {
        setError(data.message || "An error occurred on the prediction server.");
      }
    } catch (err) {
      console.error(err);
      setError("Could not connect to the backend server. Please verify that the Flask API (app.py) is running on port 5000.");
    } finally {
      setIsLoading(false);
    }
  };

  // Reset Results and Analyze Another Posting
  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <header>
        <span className="team-badge">AASTU ML GROUP 4</span>
        <h1>🔍 Recruitment Fraud Risk Analyzer</h1>
        <p>NLP-Powered Real-Time Online Job Posting Verification System</p>
      </header>

      <div className="dashboard-grid">
        {/* Left Side: Form Inputs */}
        <div className="premium-card">
          <h2>
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{ width: '22px', height: '22px' }}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" />
            </svg>
            Job Advertisement Details
          </h2>

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="title">Job Title *</label>
              <input
                id="title"
                type="text"
                name="title"
                className="form-input"
                placeholder="e.g. Executive Assistant, Data Scientist"
                value={formData.title}
                onChange={handleInputChange}
                required
                disabled={isLoading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="company_profile">Company Profile</label>
              <textarea
                id="company_profile"
                name="company_profile"
                className="form-input"
                placeholder="Brief profile of the hiring company (if provided)..."
                value={formData.company_profile}
                onChange={handleInputChange}
                disabled={isLoading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="description">Job Description</label>
              <textarea
                id="description"
                name="description"
                className="form-input"
                placeholder="Copy and paste the full job description details..."
                value={formData.description}
                onChange={handleInputChange}
                disabled={isLoading}
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="requirements">Requirements</label>
                <textarea
                  id="requirements"
                  name="requirements"
                  className="form-input"
                  placeholder="Required skills, qualifications, or certifications..."
                  value={formData.requirements}
                  onChange={handleInputChange}
                  disabled={isLoading}
                />
              </div>

              <div className="form-group">
                <label htmlFor="benefits">Benefits</label>
                <textarea
                  id="benefits"
                  name="benefits"
                  className="form-input"
                  placeholder="Salary perks, healthcare, remote options..."
                  value={formData.benefits}
                  onChange={handleInputChange}
                  disabled={isLoading}
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="employment_type">Employment Type</label>
                <select
                  id="employment_type"
                  name="employment_type"
                  className="form-input"
                  value={formData.employment_type}
                  onChange={handleInputChange}
                  disabled={isLoading}
                >
                  <option value="Full-time">Full-time</option>
                  <option value="Part-time">Part-time</option>
                  <option value="Contract">Contract</option>
                  <option value="Temporary">Temporary</option>
                  <option value="Other">Other</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="required_experience">Required Experience</label>
                <select
                  id="required_experience"
                  name="required_experience"
                  className="form-input"
                  value={formData.required_experience}
                  onChange={handleInputChange}
                  disabled={isLoading}
                >
                  <option value="Entry level">Entry level</option>
                  <option value="Mid-Senior level">Mid-Senior level</option>
                  <option value="Associate">Associate</option>
                  <option value="Director">Director</option>
                  <option value="Not Applicable">Not Applicable</option>
                </select>
              </div>
            </div>

            <div className="form-row" style={{ marginTop: '10px' }}>
              <div className="checkbox-group">
                <input
                  id="has_company_logo"
                  type="checkbox"
                  name="has_company_logo"
                  checked={formData.has_company_logo}
                  onChange={handleInputChange}
                  disabled={isLoading}
                />
                <label htmlFor="has_company_logo">Hiring Company has Logo</label>
              </div>

              <div className="checkbox-group">
                <input
                  id="has_questions"
                  type="checkbox"
                  name="has_questions"
                  checked={formData.has_questions}
                  onChange={handleInputChange}
                  disabled={isLoading}
                />
                <label htmlFor="has_questions">Includes Screening Questions</label>
              </div>
            </div>

            <div className="checkbox-group" style={{ marginBottom: '24px' }}>
              <input
                id="telecommuting"
                type="checkbox"
                name="telecommuting"
                checked={formData.telecommuting}
                onChange={handleInputChange}
                disabled={isLoading}
              />
              <label htmlFor="telecommuting">Allows Telecommuting / Fully Remote Working</label>
            </div>

            <button type="submit" className="btn-primary" disabled={isLoading}>
              {isLoading ? (
                <>
                  <div className="spinner"></div>
                  <span>Analyzing Linguistic Signatures...</span>
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{ width: '20px', height: '20px' }}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.57-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
                  </svg>
                  <span>Analyze Fraud Risk Profile</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Right Side: Risk Assessment Report */}
        <div className="premium-card" style={{ minHeight: '100%' }}>
          <h2>
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{ width: '22px', height: '22px' }}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801-12c.065.21.1.433.1.664 0 .414-.336.75-.75.75h-4.5a.75.75 0 01-.75-.75c0-.23.035-.454.1-.664m5.2 0A2.251 2.251 0 0113.5 2.25H15c1.03 0 1.9.693 2.166 1.638m-7.377 0A2.25 2.25 0 019 2.25h1.5c1.03 0 1.9.693 2.166 1.638m0 0A2.251 2.251 0 0113.5 2.25" />
            </svg>
            Risk Analysis Report
          </h2>

          {isLoading ? (
            <div className="result-placeholder">
              <div className="spinner big-spinner"></div>
              <p style={{ marginTop: '10px' }}>Extracting features & computing probabilities...</p>
            </div>
          ) : error ? (
            <div>
              <div className="error-card">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{ width: '24px', height: '24px', flexShrink: 0 }}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
                <span>{error}</span>
              </div>
              <button onClick={handleReset} className="btn-primary" style={{ background: '#3b82f6', boxShadow: 'none' }}>
                Try Again
              </button>
            </div>
          ) : result ? (
            <div>
              {/* Risk Panel Display */}
              <div className={`risk-panel ${result.label === 'HIGH RISK' ? 'high-risk' : 'low-risk'}`}>
                <div className="risk-label">{result.label}</div>
                <div className="risk-desc">
                  {result.label === 'HIGH RISK' 
                    ? "This posting displays high similarity to patterns in documented employment scam reports." 
                    : "This posting appears highly legitimate with secure linguistic features."
                  }
                </div>
              </div>

              {/* Confidence Progress Bar */}
              <div className="progress-container">
                <div className="progress-header">
                  <span>Statistical Fraud Score:</span>
                  <span className={`proba-val ${result.label === 'HIGH RISK' ? 'high' : 'low'}`}>
                    {(result.probability * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="progress-track">
                  <div 
                    className={`progress-fill ${result.label === 'HIGH RISK' ? 'high' : 'low'}`}
                    style={{ width: `${result.probability * 100}%` }}
                  ></div>
                </div>
              </div>

              {/* Identified Risk Signals List */}
              <div style={{ marginBottom: '12px', fontWeight: '500', fontFamily: 'var(--font-title)' }}>
                Identified Risk Signals ({result.signals.length})
              </div>
              
              <div className="signals-list">
                {result.signals.length > 0 ? (
                  result.signals.map((signal, idx) => (
                    <div key={idx} className="signal-card">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m0-10.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.75c0 5.592 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.57-.598-3.75h-.152c-3.196 0-6.1-1.25-8.25-3.286zm0 13.036h.008v.008H12v-.008z" />
                      </svg>
                      <span>{signal}</span>
                    </div>
                  ))
                ) : (
                  <div className="no-signals-card">
                    ✅ No major structured risk patterns identified in meta-features or description length.
                  </div>
                )}
              </div>

              <button onClick={handleReset} className="btn-primary" style={{ background: '#1e293b', border: '1px solid #334155', boxShadow: 'none', marginTop: '20px' }}>
                Analyze Another Posting
              </button>
            </div>
          ) : (
            <div className="result-placeholder">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12c0 1.268-.63 2.39-1.593 3.068a3.745 3.745 0 01-1.043 3.296 3.745 3.745 0 01-3.296 1.043A3.745 3.745 0 0112 21c-1.268 0-2.39-.63-3.068-1.593a3.746 3.746 0 01-3.296-1.043 3.745 3.745 0 01-1.043-3.296A3.745 3.745 0 013 12c0-1.268.63-2.39 1.593-3.068a3.745 3.745 0 011.043-3.296 3.746 3.746 0 013.296-1.043A3.746 3.746 0 0112 3c1.268 0 2.39.63 3.068 1.593a3.746 3.746 0 013.296 1.043 3.746 3.746 0 011.043 3.296A3.745 3.745 0 0121 12z" />
              </svg>
              <p>Enter the job posting details and click "Analyze" to extract features, compute risk probabilities, and evaluate structured warning patterns.</p>
            </div>
          )}
        </div>
      </div>

      <footer>
        <p><strong>Decision Support Framework Notice:</strong> This software is a statistical decision-support utility. Its findings must not be construed as absolute moral or legal declarations.</p>
        <p>AASTU ML Group 4 &copy; 2026. Made with ❤️ for robust recruitment security.</p>
      </footer>
    </div>
  );
}

export default App;
