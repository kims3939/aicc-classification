import React from 'react';
import { useState } from 'react';
import { makeStyles, withStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';
import FormControl from '@material-ui/core/FormControl';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import Paper from '@material-ui/core/Paper';
import LinearProgress  from '@material-ui/core/LinearProgress';
import Box from '@material-ui/core/Box';
import Typography from '@material-ui/core/Typography';
import axios from 'axios';

const useStyles = makeStyles({
  root:{
    backgroundColor:'#eeeeee'
  },

  marginBottom:{
    marginBottom:10
  }
});

const BorderLinearProgress = withStyles((theme) => ({
  root:{
    height: 10,
    borderRadius: 5
  },
  colorPrimary: {
    backgroundColor: theme.palette.grey[theme.palette.type === 'light' ? 200 : 700]
  },
  bar:{
    borderRadius: 5,
    backgroundColor: '#1a90ff'
  }
}))(LinearProgress);


const CustomLinearProgres = props => {
  return(
    <Box display='flex' alignItems='center' style={{marginBottom:10}}>
      <Box style={{marginRight:5}}>
        <Typography varient='body2'>{props.label}</Typography>
      </Box>
      <Box width='100%'>
        <BorderLinearProgress variant='determinate' value={(parseFloat(props.percent)*100).toFixed(2)}/>
      </Box>
      <Box minWidth={35} style={{marginLeft:5}}>
        <Typography variant='body2'>{(parseFloat(props.percent)*100).toFixed(2)}%</Typography>
      </Box>
    </Box>
  )
};

function App() {
  const classes = useStyles();
  const [labels, setLabels]   = useState([]);
  const [title, setTitle]     = useState('');
  const [content, setContent] = useState('')
  
  const handleClick = () => {
    axios.post('/api/v1/classification', {text:title+' '+content})
    .then(res => {
      setLabels(res.data.result);
    });
  }

  const handleTitleChange = event => {
    setTitle(event.target.value)
  }

  const handleContentChange = event => {
    setContent(event.target.value)
  }

  const labelList = labels.map(label => (
    <CustomLinearProgres key={label.category} label={label.category} percent={label.weight}/>
  ));

  return (
    <Container maxWidth='sm'>
      <Typography className={classes.marginBottom} variant='h4' align='center'>AI 상담 결과 자동분류 Demo</Typography>
      <FormControl fullWidth>
        <TextField className={classes.marginBottom} variant='outlined' label='Title' value={title} onChange={handleTitleChange}/>
        <TextField className={classes.marginBottom} variant='outlined' label='Content' multiline rows={10} value={content} onChange={handleContentChange}/>
        <Button className={classes.marginBottom} variant='contained' color='primary' onClick={handleClick}>Compute</Button>
      </FormControl>
      <Paper elevation={0}>
        {labelList}
      </Paper>
    </Container>
  );
}

export default App;
