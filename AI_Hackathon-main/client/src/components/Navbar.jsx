// src/components/Navbar.jsx
import React from "react";
import { useLocation, useNavigate } from "react-router-dom";

const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const isDashboard = location.pathname === "/dashboard";

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/");
  };

  return (
    <header className="navbar">
      <div className="navbar-left">
        <div className="navbar-logo">DocScreen</div>
      </div>
      <div className="navbar-right">
        {isDashboard ? (
          <button className="btn btn-outline" onClick={handleLogout}>
            Logout
          </button>
        ) : null}
      </div>
    </header>
  );
};

export default Navbar;
