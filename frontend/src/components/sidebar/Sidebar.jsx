import React, { useState } from 'react';
import { Box, Paper, useMediaQuery, useTheme, IconButton, Drawer, List, ListItem, ListItemIcon, ListItemText, Button, Typography, Avatar, Tooltip } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';
import MenuIcon from '@mui/icons-material/Menu';
import HomeIcon from '@mui/icons-material/Home';
import ArticleIcon from '@mui/icons-material/Article';
import PersonIcon from '@mui/icons-material/Person';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import SettingsIcon from '@mui/icons-material/Settings';
import LogoutIcon from '@mui/icons-material/Logout';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import RateReviewIcon from '@mui/icons-material/RateReview';

const Sidebar = ({ onLogout, isCollapsed, onToggleCollapse }) => {
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);

  const user = JSON.parse(localStorage.getItem('user'));
  const isAdmin = user && user.admin === true;

  const menuItems = [
    { path: '/', icon: <HomeIcon />, text: 'Home' },
    { path: '/manual', icon: <ArticleIcon />, text: 'Manual Input' },
    { path: '/linkedin', icon: <LinkedInIcon />, text: 'LinkedIn Profile' },
    { path: '/batch', icon: <UploadFileIcon />, text: 'Batch Upload' },
    { path: '/candidates', icon: <PersonIcon />, text: 'Candidates' },
    { path: '/feedback', icon: <RateReviewIcon/>, text: 'Feedback' },
    ...(isAdmin ? [{ path: '/settings', icon: <SettingsIcon />, text: 'Settings' }] : [])
  ];

  const handleDrawerToggle = () => setMobileOpen(!mobileOpen);

  const DrawerContent = ({ isMobileDrawer = false }) => (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Box sx={{p: 2, display: 'flex', alignItems: 'center', gap: 2, justifyContent: isCollapsed && !isMobileDrawer ? 'center' : 'flex-start' }}>
        <Avatar sx={{ bgcolor: '#001242', width: 40, height: 40 }}>
          {user ? user.firstName.charAt(0) : 'U'}
        </Avatar>
        {(!isCollapsed || isMobileDrawer) && (
          <Box>
            <Typography variant="subtitle1" sx={{ fontWeight: 700, color: '#222' }}>
              {user ? `${user.firstName} ${user.lastName}` : 'User'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {isAdmin ? 'Admin' : 'User'}
            </Typography>
          </Box>
        )}
      </Box>
      <List sx={{ flexGrow: 1, px: 2, mt: 2, overflowY: 'auto' }}>
        {menuItems.map((item) => (
          <ListItem
            button
            component={Link}
            to={item.path}
            key={item.text}
            onClick={isMobileDrawer ? handleDrawerToggle : undefined}
            sx={{
              mb: 1,
              borderRadius: 2,
              py: 1.2,
              px: isCollapsed && !isMobileDrawer ? 1.5 : 2,
              justifyContent: isCollapsed && !isMobileDrawer ? 'center' : 'flex-start',
              bgcolor: location.pathname === item.path ? '#EB7836' : 'transparent',
              color: location.pathname === item.path ? 'white' : '#555',
              '&:hover': {
                bgcolor: location.pathname === item.path ? '#d35400' : 'rgba(0,0,0,0.05)',
              },
            }}
          >
            <Tooltip title={isCollapsed && !isMobileDrawer ? item.text : ''} placement="right">
              <ListItemIcon sx={{ minWidth: 'auto', justifyContent: 'center', color: 'inherit', mr: (!isCollapsed || isMobileDrawer) ? 2 : 0 }}>
                {item.icon}
              </ListItemIcon>
            </Tooltip>
            {(!isCollapsed || isMobileDrawer) && (
              <ListItemText primary={item.text} sx={{ '& .MuiListItemText-primary': { fontWeight: 500, whiteSpace: 'nowrap' } }} />
            )}
          </ListItem>
        ))}
      </List>
      <Box sx={{ p: 2, mt: 'auto' }}>
        <Button
          fullWidth
          variant="text"
          onClick={onLogout}
          sx={{
            justifyContent: isCollapsed && !isMobileDrawer ? 'center' : 'flex-start',
            color: '#555',
            py: 1,
            px: 2,
            textTransform: 'none',
          }}
        >
          <Tooltip title={isCollapsed && !isMobileDrawer ? 'Logout' : ''} placement="right">
            <LogoutIcon />
          </Tooltip>
          {(!isCollapsed || isMobileDrawer) && <Typography sx={{ ml: 2 }}>Sign out</Typography>}
        </Button>
      </Box>
    </Box>
  );

  if (isMobile) {
    return (
      <>
        <Paper
          elevation={3}
          sx={{ position: 'fixed', top: 0, left: 0, right: 0, height: 'auto', bgcolor: '#fff', borderRadius: 0, zIndex: 1200, display: 'flex', alignItems: 'center', justifyContent: 'space-between', px: 2, py: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <a href="https://www.aurio.ai/de/" target="_blank" rel="noopener noreferrer">
              <img src="/static/images/logo.png" alt="Aurio Logo" style={{ height: '30px' }} />
            </a>
            <a href="https://www.uni-regensburg.de/" target="_blank" rel="noopener noreferrer">
              <img src="/static/images/ur-logo.png" alt="UR Logo" style={{ height: '30px' }} />
            </a>
          </Box>
          <IconButton color="inherit" aria-label="open drawer" edge="end" onClick={handleDrawerToggle}>
            <MenuIcon />
          </IconButton>
        </Paper>
        <Drawer
          variant="temporary"
          anchor="right"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          sx={{ '& .MuiDrawer-paper': { width: 280, boxSizing: 'border-box' } }}
        >
          <DrawerContent isMobileDrawer />
        </Drawer>
      </>
    );
  }

  return (
    <Paper
      elevation={3}
      sx={{
        position: 'fixed',
        top: 0,
        height: '100vh',
        width: isCollapsed ? 100 : 280,
        bgcolor: '#fff',
        borderRadius: 0,
        boxShadow: '0 8px 32px rgba(0,0,0,0.05)',
        zIndex: 1000,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <DrawerContent />
      <IconButton
        onClick={onToggleCollapse}
        sx={{
          position: 'absolute',
          top: 24,
          right: -12,
          bgcolor: 'white',
          border: '1px solid #e0e0e0',
          color: '#555',
          width: 24,
          height: 24,
          '&:hover': {
            bgcolor: '#f5f5f5',
          },
        }}
      >
        <ChevronLeftIcon sx={{ transform: isCollapsed ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.3s' }} />
      </IconButton>
    </Paper>
  );
};

export default Sidebar; 