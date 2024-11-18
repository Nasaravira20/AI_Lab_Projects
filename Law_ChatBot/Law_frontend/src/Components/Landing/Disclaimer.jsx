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

const Disclaimer = () => {
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
                        <span><img src='src\assets\alert.png'></img></span>
                         <u>   <h1 className="heading-black">
                                Disclaimer
                            </h1></u>
                            
                            </article>
                            <p className="text-muted lead">Welcome to LawCraftMan, your virtual assistant for initial legal guidance. Before utilizing our service, it's imperative to understand its scope and limitations. LawCraftMan is engineered to provide basic legal information and approximate cost estimates for legal services. However, it does not delve into legal loopholes or offer detailed legal strategies.</p>
                            <h3 >What LawCraftMan Can Do:</h3>
                    <article>
                        <ol>
                            <li className='text lead'><strong>Provide Section References: </strong><p className='text-muted lead'>LawCraftMan can furnish relevant legal sections and statutes based on the information provided about your case. This includes offering insights into applicable laws and regulations to help you comprehend the legal framework surrounding your issue.</p></li>
                            <li className='text lead'><strong>Offer Estimated Lawyer Fees:</strong><p className='text-muted lead'>LawCraftMan can provide approximate quotes for legal services based on the specifics of your case. These estimates are derived from typical industry rates and the complexity of the matter at hand.</p></li>   
                            <li className='text lead'><strong>Facilitate Initial Assessment:</strong><p className='text-muted lead'>LawCraftMan can assist in assessing the basic parameters of your legal situation and directing you towards appropriate legal resources or professionals for further assistance.</p></li>       
                        </ol>
                        <br/>
                        <h3 >What LawCraftMan Cannot Do:</h3>
                        <ol>
                            <li className='text lead'><strong>Explore Legal Loopholes: </strong><p className='text-muted lead'>LawCraftMan does not analyze or exploit legal loopholes. It is designed to provide straightforward information and guidance within the bounds of established legal principles.</p></li>
                            <li className='text lead'><strong>Offer Detailed Legal Strategies:</strong><p className='text-muted lead'> LawCraftMan does not provide in-depth legal strategies or tactics. For comprehensive legal advice tailored to your specific circumstances, it's advisable to consult with a qualified attorney.</p></li>   
                            <li className='text lead'><strong>Guarantee Exact Cost Estimates:</strong><p className='text-muted lead'>While LawCraftMan offers approximate cost estimates for legal services, it cannot guarantee exact figures. Actual fees may vary depending on factors such as the complexity of the case, attorney expertise, and regional differences.</p></li>
                            </ol>
                            <br/>
                        <p className="col-md-20 mx-auto text-center text-muted lead">While LawCraftMan offers approximate cost estimates for legal services, it cannot guarantee exact figures. Actual fees may vary depending on factors such as the complexity of the case, attorney expertise, and regional differences.</p>                
                        </article>
                        <article className="col-md-20 mx-auto text-center">
                         <h1 className="heading-black">
                         
                         "LawCraftMan: Guiding You Towards Justice, Not Judgment."
                            </h1>
                            </article>

                    </article>
                </article>
    
                
    
            </section>
        </main>
      )
}

export default Disclaimer