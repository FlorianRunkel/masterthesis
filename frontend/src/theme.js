import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#001B41', // aurio-blue
      light: '#FF5F00', // aurio-orange
    },
    background: {
      default: 'rgba(242, 242, 242, 0.6)',
      paper: '#FFFFFF',
    },
    text: {
      primary: '#333333',
      secondary: '#666666',
    },
    error: {
      main: '#dc3545', // error-red
    },
    warning: {
      main: '#ffc107', // warning-yellow
    },
    success: {
      main: '#28a745', // success-green
    }
  },
  typography: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      color: '#001B41',
      letterSpacing: '-0.5px',
      marginBottom: '20px',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      color: '#001B41',
      marginBottom: '20px',
    },
    h6: {
      fontSize: '1.5rem',
      fontWeight: 600,
      color: '#001B41',
    },
    body1: {
      fontSize: '1.1rem',
      color: '#666666',
      lineHeight: 1.5,
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: 'rgba(242, 242, 242, 0.6)',
          '&::before': {
            content: '""',
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            background: 'radial-gradient(circle, rgba(245, 245, 240, 0) 40%, rgba(245, 245, 240, 0.8) 70%, rgba(242, 242, 242, 1) 100%)',
            zIndex: -1,
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        contained: {
          backgroundColor: '#001B41',
          color: '#FFFFFF',
          padding: '14px',
          fontSize: '1rem',
          fontWeight: 600,
          borderRadius: '8px',
          '&:hover': {
            backgroundColor: '#FF5F00',
            transform: 'translateY(-2px)',
          },
          transition: 'all 0.3s ease',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: '10px',
            '& fieldset': {
              borderColor: '#ddd',
            },
            '&:hover fieldset': {
              borderColor: '#001B41',
            },
            '& input': {
              fontSize: '1rem',
              padding: '12px',
            },
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: '0 8px 25px rgba(0, 0, 0, 0.1)',
          borderRadius: '16px',
          padding: '30px',
        },
      },
    },
    MuiListItem: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          marginBottom: '15px',
          transition: 'background-color 0.2s',
          '&:hover': {
            backgroundColor: 'rgba(242, 242, 242, 0.6)',
          },
          '&.Mui-selected': {
            backgroundColor: '#001B41',
            color: '#FFFFFF',
            '&:hover': {
              backgroundColor: '#FF5F00',
            },
          },
        },
      },
    },
  },
});

export default theme; 