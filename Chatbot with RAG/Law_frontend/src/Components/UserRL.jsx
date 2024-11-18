import React from 'react'
import Navbar from './Landing/Navbar'
import Hero from './Landing/Hero'
import Theme from './Landing/Theme'
import Features from './Landing/Features'
import Disclaimer from './Landing/Disclaimer'
import FAQ from './Landing/FAQ'
const UserRL = () => {
  return (
    <div className='h-screen' >
        <section >
      <section >
        <Navbar />  
        <Hero />
        <Features/>
        <Disclaimer/>
        <FAQ/>
      </section>
      <Theme />
      </section>
    </div>
  )
}

export default UserRL
