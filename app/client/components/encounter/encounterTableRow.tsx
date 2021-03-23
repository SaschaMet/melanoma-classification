import React from 'react';
import { useQuery } from "@apollo/client";
import Link from 'next/link'

import { ENCOUNTER } from "../../graphQl/queries/encounter"

import { Encounter } from "../../types/fhir"


export default function TableRow ({ encounterId, index }) {

    const { loading, error, data } = useQuery(ENCOUNTER, {variables: { id: encounterId }});

    if (loading || error) return null;

    const encounter = data.Encounter as Encounter

    return (
        <tr>
            <th scope="row">{index + 1}</th>
            <td>{encounter.meta.lastUpdated}</td>
            <td>
                <Link href={`/encounter/${encounter.id}`}>
                    <button type="button" className="btn btn-secondary btn-sm">Details</button>
                </Link>
            </td>
        </tr>
    )
}