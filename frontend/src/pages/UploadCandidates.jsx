import React, { useState } from 'react';
import { Tabs, Tab, Box } from '@mui/material';
import ManualInput from '../components/candidates/ManualInput';
import LinkedInInput from '../components/candidates/LinkedInInput';
import BatchUpload from '../components/candidates/BatchUpload';

const UploadCandidates = () => {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <div className="upload-candidates">
      <div className="header-section">
        <h1>Kandidaten Analyse</h1>
        <p className="info-description">
          Wählen Sie eine Methode, um Kandidaten für die Karriereanalyse hochzuladen.
        </p>
      </div>

      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={activeTab} onChange={handleTabChange}>
          <Tab label="Manuelle Eingabe" />
          <Tab label="LinkedIn Import" />
          <Tab label="Batch Upload" />
        </Tabs>
      </Box>

      <div className="tab-content">
        {activeTab === 0 && <ManualInput />}
        {activeTab === 1 && <LinkedInInput />}
        {activeTab === 2 && <BatchUpload />}
      </div>
    </div>
  );
};

export default UploadCandidates; 