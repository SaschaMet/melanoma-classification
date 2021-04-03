import React from 'react';
import { useQuery } from "@apollo/client";

import TableRow from "./encounterTableRow"
import { ENCOUNTERS_BY_PATIENT_ID } from "../../graphQl/queries/encounter"

export default function EncounterTable ({patientId}) {

    const { loading, error, data } = useQuery(ENCOUNTERS_BY_PATIENT_ID, {variables: { patientId }});

    if (loading) return <div className="col-4">Loading ...</div>;
    if (error) return null;

    const encounters = data.EncounterByPatientId.entries

    if(!encounters || !encounters.length) {
        return (
            <div className="mt-5">
                There are no encounters for this patient.
            </div>
        )
    }

    return (
        <table className="table table-striped mt-5">
            <thead>
                <tr>
                    <th scope="col">#</th>
                    <th scope="col">Date</th>
                    <th scope="col">Details</th>
                </tr>
            </thead>
            <tbody>
                {encounters.map((encounterId, i) => <TableRow key={encounterId} encounterId={encounterId} index={i} />)}
            </tbody>
        </table>
    );
}
