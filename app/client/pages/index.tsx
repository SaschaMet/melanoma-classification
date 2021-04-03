import React from 'react'
import { useQuery } from "@apollo/client";

import { ALL_PATIENTS_QUERY } from "../graphQl/queries/patient"

import Patient from "../components/patient/overview"

export default function Home() {

    const { loading, error, data } = useQuery(ALL_PATIENTS_QUERY);

    if (error) return <div>Error</div>;
    if (loading) return <div>Loading ...</div>;

    const patientIds = data.AllPatientsQuery.entries

    return (
        <div className="container-fluid">
            <div className="row">
                <div className="col-xs-12 mb-5">
                    <h1>
                        Dashboard
                    </h1>
                </div>
                {patientIds.map(patient => <Patient key={patient} id={patient} /> )}
            </div>
        </div>
    )
}
