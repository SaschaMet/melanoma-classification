import React from 'react';
import Head from 'next/head';
import { ApolloProvider } from '@apollo/client';
import Sidebar from '../components/sidebar';
import { useApollo } from '../utils/apolloClient';

import 'bootstrap/dist/css/bootstrap.css';
import '../styles/custom.css';

function MelanomaClassification ({ Component, pageProps, }) {
    const apolloClient = useApollo(pageProps.initialApolloState || {});

    return (
        <React.Fragment>
            <ApolloProvider client={apolloClient}>
                <Head>
                    <title>Melanoma Classification</title>
                    <meta name="viewport" content="initial-scale=1.0, width=device-width" />
                </Head>
                <div className="container-fluid" style={{ height: '100vh', }}>
                    <div className="row h-100">
                        <Sidebar />
                        <div className="col ps-5 pt-5">
                            <Component {...pageProps} />
                        </div>
                    </div>
                </div>
            </ApolloProvider>
        </React.Fragment>
    );
}

export default MelanomaClassification;
