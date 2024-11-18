import React from 'react'
import { CiEdit } from "react-icons/ci";
import { FaLanguage } from "react-icons/fa";
import { GrResources } from "react-icons/gr";
import { FcDisclaimer } from "react-icons/fc";
import { lawyer,multiple,guid,cost } from '../../assets';

import Slider from 'react-slick';
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';
import { useState, useEffect } from 'react';

const FAQ = () => {
    const [scrollTop, setScrollTop] = useState(0);
    const [isScrolling, setIsScrolling] = useState(false);
    const [isUp, setIsUp] = useState(false);
   
    useEffect(() => {
       const onScroll = () => {
         const scroll = window.scrollY;
         setScrollTop(scroll);
         setIsScrolling(scroll > 80);
         setIsUp(scroll > scrollTop);
       };
   
       window.addEventListener('scroll', onScroll);
       return () => window.removeEventListener('scroll', onScroll);
    }, [scrollTop]);
   
    const settings = {
       dots: true,
       infinite: true,
       speed: 500,
       slidesToShow: 1,
       slidesToScroll: 1,
       autoplay: true,
       autoplaySpeed: 3000,
    };
    return (
        <main>
            <section className="pt-6 pb-7 overflow-hidden" id="features">
                <article className="container">
                
                    <article className="row">
                    <article className="col-md-8 mx-auto text-center">
                    <u>   <h1 className="heading-black">
                                FAQs
                            </h1></u>
                            </article>
                        <ol>
                            <li className='text lead'><strong>What is LawCraftMan?</strong><p className='text-muted lead'>LawCraftMan is an innovative law chatbot designed to provide initial legal guidance and assistance to individuals facing legal issues. It offers basic information, section references, estimated lawyer fees, and direction towards relevant legal resources.</p></li>
                            <li className='text lead'><strong>How does LawCraftMan work?</strong><p className='text-muted lead'>LawCraftMan operates through a user-friendly interface where you can input details about your legal situation. Based on the information provided, it generates relevant legal sections, offers approximate lawyer fees, and provides general guidance to help you understand your rights and options.</p></li>   
                            <li className='text lead'><strong>Can LawCraftMan provide personalized legal advice?</strong><p className='text-muted lead'>No, LawCraftMan cannot offer personalized legal advice. While it can provide general information and estimates, it is not a substitute for consultation with a qualified attorney. For tailored legal advice specific to your circumstances, it's essential to consult with a licensed legal professional.</p></li>
                            <li className='text lead'><strong>What types of legal issues can LawCraftMan assist with?</strong><p className='text-muted lead'>LawCraftMan can provide guidance on a wide range of legal topics, including but not limited to family law, contracts, employment law, property disputes, and more. However, it is important to note that its scope is limited to initial assistance and general information provision.</p></li>
                            <li className='text lead'><strong>How accurate are the estimated lawyer fees provided by LawCraftMan?</strong><p className='text-muted lead'>The estimated lawyer fees offered by LawCraftMan are based on typical industry rates and the complexity of the case provided. While they aim to provide a rough estimate, actual fees may vary depending on various factors. It's advisable to confirm fees directly with the attorney or law firm.</p></li>       
                        </ol>
                        <br/> 
                            <p className="text-muted lead"></p>
                        </article>
                    </article>
                
    
                
    
            </section>
        </main>
      )
}

export default FAQ