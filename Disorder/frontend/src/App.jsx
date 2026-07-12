import React, { useState, useMemo, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, AlertTriangle, Users, Filter, Download, Search, ChevronDown, ChevronUp, Info, Zap, RotateCcw } from 'lucide-react';

// --- Defines the features a business can actually change ---
const ACTIONABLE_LEVERS = new Set([
  'CURR_ANN_AMT',
  'AMT_PER_DAY_TENURE',
  'PREMIUM_TO_INCOME_RATIO'
]);

const ChurnPredictionUI = () => {
  // --- STATE HOOKS ---
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [filterRisk, setFilterRisk] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCustomer, setSelectedCustomer] = useState(null);
  const [sortBy, setSortBy] = useState('churn_probability');
  const [sortOrder, setSortOrder] = useState('desc');

  // --- INDIVIDUAL SIMULATION STATE ---
  const [simulationData, setSimulationData] = useState(null);
  const [simulationResult, setSimulationResult] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [topSimulationFeatures, setTopSimulationFeatures] = useState([]);

  // --- GLOBAL SIMULATION STATE ---
  const [policyChange, setPolicyChange] = useState({ percentage: 0 });
  const [isGlobalSimulating, setIsGlobalSimulating] = useState(false);
  const [analysisType, setAnalysisType] = useState('Original');

  // --- EFFECT for Individual Customer Simulation ---
  useEffect(() => {
    if (selectedCustomer) {
      // Find top 2 features that are BOTH risk-increasing AND actionable
      const actionableRiskFactors = selectedCustomer.feature_contributions
        .filter(feat => feat.impact === 'increases risk' && ACTIONABLE_LEVERS.has(feat.feature))
        .slice(0, 2);
      
      setTopSimulationFeatures(actionableRiskFactors);
  
      // Reconstruct the full data payload, ensuring base features for calculation are present
      const fullCustomerData = selectedCustomer.feature_contributions.reduce((acc, feat) => {
        let value = String(feat.feature_value).replace(/,/g, '');
        acc[feat.feature] = parseFloat(value);
        return acc;
      }, {});
      
      // Add defaults for base features if they're not in the contributions list,
      // which is crucial for recalculations.
      if (!fullCustomerData.hasOwnProperty('INCOME')) {
        const incomeFeature = selectedCustomer.feature_contributions.find(f => f.feature === 'INCOME');
        fullCustomerData['INCOME'] = incomeFeature ? parseFloat(String(incomeFeature.feature_value).replace(/,/g, '')) : 50000;
      }
       if (!fullCustomerData.hasOwnProperty('DAYS_TENURE')) {
        const tenureFeature = selectedCustomer.feature_contributions.find(f => f.feature === 'DAYS_TENURE');
        fullCustomerData['DAYS_TENURE'] = tenureFeature ? parseFloat(String(tenureFeature.feature_value).replace(/,/g, '')) : 365;
      }
      
      setSimulationData(fullCustomerData);
      setSimulationResult(null);
    }
  }, [selectedCustomer]);


  // --- THIS IS THE CORRECTED HANDLER WITH INDEPENDENT SLIDERS ---
  const handleSimulationValueChange = (feature, value) => {
    setSimulationData(prev => {
      // Create a new data object with the updated slider value
      const newData = { ...prev, [feature]: value };
      const epsilon = 1e-6; // To prevent division by zero

      // **REVISED LOGIC**: ONLY perform forward calculation. No back-calculation.
      // This keeps the sliders independent from the user's perspective.
      if (feature === 'CURR_ANN_AMT') {
        if (newData.hasOwnProperty('PREMIUM_TO_INCOME_RATIO')) {
          newData['PREMIUM_TO_INCOME_RATIO'] = value / (newData['INCOME'] + epsilon);
        }
        if (newData.hasOwnProperty('AMT_PER_DAY_TENURE')) {
          newData['AMT_PER_DAY_TENURE'] = value / (newData['DAYS_TENURE'] + epsilon);
        }
      }
      
      // The back-calculation logic that was here has been removed to keep sliders independent.

      return newData;
    });
  };

  const runIndividualSimulation = async () => {
    if (!simulationData) return;
    setIsSimulating(true);
    setSimulationResult(null);
    try {
      const response = await fetch('http://127.0.0.1:5000/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(simulationData),
      });
      if (!response.ok) throw new Error('Simulation request failed');
      const result = await response.json();
      setSimulationResult(result);
    } catch (error) {
      console.error('Simulation error:', error);
      alert('Failed to run individual simulation. Is the backend server running?');
    } finally {
      setIsSimulating(false);
    }
  };

  const fetchPredictionsFromServer = async (policy = null) => {
    setLoading(true);
    setIsGlobalSimulating(true);
    setPredictions(null);
    
    try {
      const response = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ policy }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const jsonData = await response.json();

      const totalCustomers = jsonData.length;
      const avgChurnProb = jsonData.reduce((sum, c) => sum + c.churn_probability, 0) / totalCustomers;
      const highRiskCount = jsonData.filter(c => c.risk_level === 'HIGH').length;
      const moderateRiskCount = jsonData.filter(c => c.risk_level === 'MODERATE').length;
      const lowRiskCount = jsonData.filter(c => c.risk_level === 'LOW').length;

      setPredictions({
        summary: { totalCustomers, avgChurnProb, highRiskCount, moderateRiskCount, lowRiskCount, highRiskPercent: (highRiskCount / totalCustomers) * 100 },
        customers: jsonData
      });
      
      setAnalysisType(policy ? `Simulated (${policy.percentage > 0 ? '+' : ''}${policy.percentage}%)` : 'Original');

    } catch (error) {
      console.error('Error fetching predictions:', error);
      alert('Could not connect to the analysis server. Please ensure the Python backend is running.');
    } finally {
      setLoading(false);
      setIsGlobalSimulating(false);
    }
  };

  const handleApplyPolicy = () => {
    const policy = { feature: 'CURR_ANN_AMT', percentage: policyChange.percentage };
    fetchPredictionsFromServer(policy);
  };

  const handleResetPolicy = () => {
    setPolicyChange({ percentage: 0 });
    fetchPredictionsFromServer(null);
  };

  // Memoized calculations
  const filteredCustomers = useMemo(() => {
    if (!predictions) return [];
    let filtered = predictions.customers.filter(c => (filterRisk === 'all' || c.risk_level.toLowerCase() === filterRisk) && (searchQuery === '' || c.customer_id.toString().toLowerCase().includes(searchQuery.toLowerCase())));
    filtered.sort((a, b) => { let aVal = a[sortBy], bVal = b[sortBy]; return sortOrder === 'asc' ? (aVal > bVal ? 1 : -1) : (aVal < bVal ? 1 : -1); });
    return filtered;
  }, [predictions, filterRisk, searchQuery, sortBy, sortOrder]);

  const distributionData = useMemo(() => {
    if (!predictions) return [];
    const buckets = [ { range: '0-20%', count: 0, fill: '#22c55e' }, { range: '20-40%', count: 0, fill: '#84cc16' }, { range: '40-60%', count: 0, fill: '#eab308' }, { range: '60-80%', count: 0, fill: '#f97316' }, { range: '80-100%', count: 0, fill: '#ef4444' }];
    predictions.customers.forEach(c => { const prob = c.churn_probability * 100; if (prob < 20) buckets[0].count++; else if (prob < 40) buckets[1].count++; else if (prob < 60) buckets[2].count++; else if (prob < 80) buckets[3].count++; else buckets[4].count++; });
    return buckets;
  }, [predictions]);
  
  const riskLevelData = useMemo(() => { if (!predictions) return []; return [ { name: 'High', value: predictions.summary.highRiskCount, fill: '#ef4444' }, { name: 'Moderate', value: predictions.summary.moderateRiskCount, fill: '#f97316' }, { name: 'Low', value: predictions.summary.lowRiskCount, fill: '#22c55e' } ]; }, [predictions]);
  const reasonData = useMemo(() => { if (!predictions) return []; const reasons = {}; predictions.customers.forEach(c => { if (c.feature_contributions && c.feature_contributions.length > 0) { const reason = c.feature_contributions[0].feature.replace(/_/g, ' '); reasons[reason] = (reasons[reason] || 0) + 1; } }); return Object.entries(reasons).map(([name, value]) => ({ name, value })).sort((a, b) => b.value - a.value).slice(0, 5); }, [predictions]);
  const exportResults = () => { const dataStr = JSON.stringify(predictions, null, 2); const dataBlob = new Blob([dataStr], { type: 'application/json' }); const url = URL.createObjectURL(dataBlob); const link = document.createElement('a'); link.href = url; link.download = 'churn_predictions.json'; link.click(); };

  // --- RENDER FUNCTION ---
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 text-slate-800">
      <div className="bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-semibold text-slate-900">Churn Prediction Dashboard</h1>
            <p className="text-sm text-slate-600 mt-1">Analyze customer churn risk and take proactive action</p>
          </div>
          {predictions && <div className="text-right"><p className="text-xs text-slate-500">Analysis Type</p><p className="font-semibold text-blue-600">{analysisType}</p></div>}
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {!predictions && !loading && ( <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-12 text-center"><TrendingUp className="w-16 h-16 text-slate-400 mx-auto mb-4" /><h2 className="text-xl font-semibold text-slate-900 mb-2">Ready to Analyze</h2><p className="text-slate-600 mb-6">Click the button to load and analyze the original customer data.</p><button onClick={() => fetchPredictionsFromServer(null)} className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer transition">Load Original Data</button></div> )}
        {loading && ( <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-12 text-center"><div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div><p className="text-slate-600">Connecting to AI engine and analyzing data...</p></div> )}
        
        {predictions && !loading && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
              <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2"><Zap className="w-5 h-5 text-blue-600"/>Global Retention Policy Simulator</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-slate-700">Adjust Annual Premium for All Customers</label>
                  <div className="flex items-center gap-4 mt-2">
                    <span className="text-sm font-semibold text-red-600">-20%</span>
                    <input type="range" min="-20" max="20" step="1" value={policyChange.percentage} onChange={(e) => setPolicyChange({ percentage: parseInt(e.target.value) })} className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"/>
                    <span className="text-sm font-semibold text-green-600">+20%</span>
                  </div>
                </div>
                <div className="flex flex-col gap-2">
                  <span className="text-center text-2xl font-bold text-blue-600">{policyChange.percentage > 0 ? '+' : ''}{policyChange.percentage}%</span>
                  <div className="flex gap-2">
                     <button onClick={handleApplyPolicy} disabled={isGlobalSimulating || policyChange.percentage === 0} className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:bg-slate-400">Apply</button>
                     <button onClick={handleResetPolicy} disabled={isGlobalSimulating || analysisType === 'Original'} className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition disabled:bg-slate-400"><RotateCcw className="w-4 h-4"/>Reset</button>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6 flex items-center justify-between"><Users className="w-8 h-8 text-blue-600" /><div className="text-right"><p className="text-sm text-slate-600">Total Customers</p><p className="text-2xl font-bold text-slate-900">{predictions.summary.totalCustomers}</p></div></div>
              <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6 flex items-center justify-between"><TrendingUp className="w-8 h-8 text-green-600" /><div className="text-right"><p className="text-sm text-slate-600">Avg Churn Probability</p><p className="text-2xl font-bold text-slate-900">{(predictions.summary.avgChurnProb * 100).toFixed(1)}%</p></div></div>
              <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6 flex items-center justify-between"><AlertTriangle className="w-8 h-8 text-red-600" /><div className="text-right"><p className="text-sm text-slate-600">High Risk Customers</p><p className="text-2xl font-bold text-slate-900">{predictions.summary.highRiskCount}</p></div></div>
              <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6 flex items-center justify-between"><Filter className="w-8 h-8 text-orange-600" /><div className="text-right"><p className="text-sm text-slate-600">High Risk Percentage</p><p className="text-2xl font-bold text-slate-900">{predictions.summary.highRiskPercent.toFixed(1)}%</p></div></div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6"><h3 className="text-lg font-semibold text-slate-900 mb-4">Churn Probability Distribution</h3><ResponsiveContainer width="100%" height={250}><BarChart data={distributionData}><CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" /><XAxis dataKey="range" stroke="#64748b" /><YAxis stroke="#64748b" /><Tooltip /><Bar dataKey="count" radius={[4, 4, 0, 0]}>{distributionData.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.fill} />)}</Bar></BarChart></ResponsiveContainer></div>
              <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6"><h3 className="text-lg font-semibold text-slate-900 mb-4">Risk Level Distribution</h3><ResponsiveContainer width="100%" height={250}><PieChart><Pie data={riskLevelData} cx="50%" cy="50%" labelLine={false} outerRadius={80} dataKey="value" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>{riskLevelData.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.fill} />)}</Pie><Tooltip /></PieChart></ResponsiveContainer></div>
              <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6"><h3 className="text-lg font-semibold text-slate-900 mb-4">Top Churn Risk Factors</h3><ResponsiveContainer width="100%" height={250}><BarChart data={reasonData} layout="vertical"><CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" /><XAxis type="number" stroke="#64748b" /><YAxis dataKey="name" type="category" stroke="#64748b" width={120} /><Tooltip /><Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} /></BarChart></ResponsiveContainer></div>
            </div>

            <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
              <div className="flex flex-wrap items-center gap-4 mb-6"><div className="flex-1 min-w-[200px] relative"><Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" /><input type="text" placeholder="Search by Customer ID..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} className="w-full pl-10 pr-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" /></div><select value={filterRisk} onChange={(e) => setFilterRisk(e.target.value)} className="px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"><option value="all">All</option><option value="high">High</option><option value="moderate">Moderate</option><option value="low">Low</option></select><button onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')} className="flex items-center gap-2 px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 transition">{sortOrder === 'asc' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}{sortOrder === 'asc' ? 'Asc' : 'Desc'}</button><button onClick={exportResults} className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition"><Download className="w-5 h-5" /> Export</button></div>
              <div className="overflow-x-auto"><table className="w-full"><thead className="bg-slate-50 border-b border-slate-200"><tr><th className="px-4 py-3 text-left text-xs font-medium text-slate-600 uppercase">Customer ID</th><th className="px-4 py-3 text-left text-xs font-medium text-slate-600 uppercase">Churn Risk</th><th className="px-4 py-3 text-left text-xs font-medium text-slate-600 uppercase">Risk Level</th><th className="px-4 py-3 text-left text-xs font-medium text-slate-600 uppercase">Top Factor</th><th className="px-4 py-3 text-left text-xs font-medium text-slate-600 uppercase">Action</th></tr></thead><tbody className="divide-y divide-slate-200">{filteredCustomers.slice(0, 50).map((c) => (<tr key={c.customer_id} className="hover:bg-slate-50"><td className="px-4 py-3 text-sm font-medium text-slate-900">{c.customer_id}</td><td className="px-4 py-3"><div className="flex items-center gap-2"><div className="flex-1 bg-slate-200 rounded-full h-2 max-w-[100px]"><div className={`h-2 rounded-full ${c.risk_level === 'HIGH' ? 'bg-red-600' : c.risk_level === 'MODERATE' ? 'bg-orange-500' : 'bg-green-500'}`} style={{ width: `${c.churn_probability * 100}%` }} /></div><span className="text-sm font-semibold text-slate-900">{(c.churn_probability * 100).toFixed(1)}%</span></div></td><td className="px-4 py-3"><span className={`text-xs px-2 py-1 rounded font-medium ${c.risk_level === 'HIGH' ? 'bg-red-100 text-red-700' : c.risk_level === 'MODERATE' ? 'bg-orange-100 text-orange-700' : 'bg-green-100 text-green-700'}`}>{c.risk_level}</span></td><td className="px-4 py-3 text-sm text-slate-700">{c.feature_contributions?.[0]?.feature.replace(/_/g, ' ') || 'N/A'}</td><td className="px-4 py-3"><button onClick={() => setSelectedCustomer(c)} className="text-sm text-blue-600 hover:text-blue-800 font-medium">Details</button></td></tr>))}</tbody></table></div>
            </div>
          </div>
        )}

        {/* --- CUSTOMER DETAIL MODAL (Reverted to Top 2 Logic with Bug Fix) --- */}
        {selectedCustomer && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4" onClick={() => setSelectedCustomer(null)}>
            <div className="bg-white rounded-xl shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto p-8" onClick={(e) => e.stopPropagation()}>
              <h3 className="text-2xl font-bold text-slate-900 mb-6">Customer Details: {selectedCustomer.customer_id}</h3>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                <div><p className="text-sm text-slate-600 mb-1">Original Probability</p><p className="text-lg font-semibold text-red-600">{(selectedCustomer.churn_probability * 100).toFixed(1)}%</p></div>
                <div><p className="text-sm text-slate-600 mb-1">Risk Level</p><span className="inline-block text-sm px-3 py-1 rounded font-medium bg-red-100 text-red-700">{selectedCustomer.risk_level}</span></div>
                {simulationResult && (<><div><p className="text-sm text-slate-600 mb-1">Simulated Probability</p><p className="text-lg font-semibold text-green-600">{(simulationResult.churn_probability * 100).toFixed(1)}%</p></div><div><p className="text-sm text-slate-600 mb-1">Change</p><p className={`text-lg font-semibold ${simulationResult.churn_probability < selectedCustomer.churn_probability ? 'text-green-600' : 'text-red-600'}`}>{(selectedCustomer.churn_probability - simulationResult.churn_probability > 0 ? '-' : '+')}{(Math.abs(selectedCustomer.churn_probability - simulationResult.churn_probability) * 100).toFixed(1)}%</p></div></>)}
              </div>
              
              <div className="p-4 bg-slate-50 rounded-lg mb-6 border border-slate-200">
                <h4 className="font-semibold text-slate-900 mb-4">⚙️ Retention Strategy Simulator</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {topSimulationFeatures.length > 0 ? (
                    topSimulationFeatures.map(featureInfo => {
                      let min = 500, max = 10000, step = 100; 
                      if(featureInfo.feature.includes('RATIO')){
                          min = 0; max = 1; step = 0.05;
                      } else if (featureInfo.feature.includes('AMT_PER_DAY_TENURE')) {
                          min = 0; max = 50; step = 1;
                      }
                      return (
                        <div key={featureInfo.feature}>
                          <label className="block text-sm font-medium text-slate-700 capitalize">
                            {featureInfo.feature.replace(/_/g, ' ')}
                          </label>
                          <input
                            type="range"
                            min={min}
                            max={max}
                            step={step}
                            value={simulationData[featureInfo.feature] || 0}
                            onChange={(e) => handleSimulationValueChange(featureInfo.feature, parseFloat(e.target.value))}
                            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                          />
                           <span className="text-sm text-slate-500">
                              {featureInfo.feature.includes('AMT') && !featureInfo.feature.includes('RATIO') ? '$' : ''}
                              { (simulationData[featureInfo.feature] || 0).toFixed(2) }
                            </span>
                        </div>
                      );
                    })
                  ) : (
                    <p className="text-sm text-slate-600 col-span-2">
                      This customer's risk is primarily driven by non-actionable factors. No direct levers to simulate.
                    </p>
                  )}
                </div>
                <button onClick={runIndividualSimulation} disabled={isSimulating || topSimulationFeatures.length === 0} className="mt-4 w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:bg-slate-400">
                  {isSimulating ? 'Simulating...' : 'Run Simulation'}
                </button>
              </div>
              
              <div className="mb-6">
                <h4 className="font-semibold text-slate-900 mb-3 flex items-center gap-2"><Info className="w-5 h-5" /> AI Reasoning</h4>
                <p className="text-slate-700 bg-slate-100 p-4 rounded-lg">{simulationResult?.reasoning || selectedCustomer.reasoning}</p>
              </div>
              <div className="p-4 bg-green-50 rounded-lg mb-6 border border-green-200">
                <h4 className="font-semibold text-green-900 mb-2">Retention Recommendations</h4>
                <ul className="text-sm text-green-800 space-y-1">{(simulationResult?.retention_recommendations || selectedCustomer.retention_recommendations)?.map((rec, idx) => <li key={idx}>• {rec}</li>)}</ul>
              </div>
              <div className="mb-6">
                <h4 className="font-semibold text-slate-900 mb-3">Top Feature Contributions</h4>
                <div className="space-y-2">
                  {selectedCustomer.feature_contributions?.slice(0, 5).map((feature, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                      <div>
                        <p className="text-sm font-medium text-slate-900">{feature.feature.replace(/_/g, ' ')}</p>
                        <p className="text-xs text-slate-600">Value: {feature.feature_value}</p>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className={`text-xs px-2 py-1 rounded ${feature.impact === 'increases risk' ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>{feature.impact}</span>
                        <span className="text-sm font-semibold text-slate-900">{feature.shap_value > 0 ? '+' : ''}{feature.shap_value.toFixed(4)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <button onClick={() => setSelectedCustomer(null)} className="w-full px-4 py-2 bg-slate-200 text-slate-700 rounded-lg hover:bg-slate-300 transition">Close</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChurnPredictionUI;

