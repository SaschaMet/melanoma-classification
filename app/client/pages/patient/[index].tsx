import React from 'react';
import { useQuery } from "@apollo/client";
import Link from 'next/link'

import EncounterTable from "../../components/encounter/encounterTable"

import { PATIENT } from "../../graphQl/queries/patient"
import { Patient } from "../../types/fhir"


export default function PatientOverview (props) {

    if(typeof window === 'undefined') {
        return null
    }

    // get the patient id from the url
    const id = window.location.pathname.split("/patient/")[1]

    const { loading, error, data } = useQuery(PATIENT, {variables: { id }});

    if (loading) return <div className="col-4">Loading ...</div>;
    if (error) return null;


    const patient = data.Patient as Patient

    return (
        <div className="col-12 row">
            <div className="col-8 row">
                <div className="col">
                    <img
                        width="140"
                        height="140"
                        className="rounded-circle"
                        src={patient.gender === "female" ? "/images/female-avatar.png" : "/images/male-avatar.png"}
                    />
                    <p className="card-title h5 my-4">Name: {patient.name[0].text}</p>
                    <p className="card-text text-start">Gender: {patient.gender}</p>
                </div>
            </div>
            <div className="col-12">
                <EncounterTable patientId={id} />
            </div>
        </div>
    );
}
