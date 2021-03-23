import React from 'react';
import { useQuery } from "@apollo/client";
import Link from 'next/link'

import { PATIENT } from "../../graphQl/queries/patient"
import { Patient } from "../../types/fhir"


export default function PatientOverview ({ id, }) {
    const { loading, error, data } = useQuery(PATIENT, {variables: { id }});

    if (loading) return <div className="col-4">Loading ...</div>;
    if (error) return null;


    const patient = data.Patient as Patient

    return (
        <div className="col-4">
            <div className="card">
                <div className="card-body">
                    <div className="text-center">
                        <img
                            width="140"
                            height="140"
                            className="rounded-circle"
                            src={patient.gender === "female" ? "/images/female-avatar.png" : "/images/male-avatar.png"}
                        />
                    </div>
                    <p className="card-title h5 my-5">{patient.name[0].text}</p>
                    <p className="card-text text-left">Gender: {patient.gender}</p>
                    <Link href={"/patient/"+id} passHref>
                        <a className="card-link btn btn-secondary">Patient Details</a>
                    </Link>
                </div>
            </div>
        </div>
    );
}
