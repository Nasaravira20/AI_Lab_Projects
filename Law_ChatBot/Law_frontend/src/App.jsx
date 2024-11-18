//import { useState } from 'react'
//import './App.css'

import {Routes,Route} from 'react-router-dom'
import UserRL from './Components/UserRL'
import Features from './Components/Landing/Features'
import NotFound from './Components/NotFound'
import Disclaimer from './Components/Landing/Disclaimer'
import Home from './Components/Chatbot/Chatbot'



function App() {
  //const [count, setCount] = useState(0)
  

  return (
    < >
    
      
    <Routes>
      <Route exact path='/' element={<UserRL/>}/>
      <Route path='/features' element={<Features/>}/>
      <Route path='*' element={<NotFound/>}/>
      <Route path='/disclaimer' element={<Disclaimer/>}/>
      <Route path='/Home' element={<Home/>}/>
    </Routes> 
     
    
    </>
    

  )
}

export default App
