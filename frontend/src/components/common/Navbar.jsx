import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="nav-container">
        <div className="logo-container">
          <div className="logo-wrapper">
            <img src="/static/images/logo.png" alt="Aurio Technology Logo" className="navbar-logo" />
            <img src="/static/images/ur-logo.png" alt="UR Logo" className="navbar-logo" />
          </div>
        </div>
        <div className="nav-menu">
          <div className="nav-item">
            <img src="/static/images/carrer_nav.jpeg" alt="Karriere Navigation" className="nav-icon" />
            <Link to="/">Kandidaten Analyse</Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 