import React from "react"
import { useQuery } from "@apollo/client";

import { Media } from "../../types/fhir"
import { MEDIA } from "../../graphQl/queries/media"

export default function MediaDetail({ mediaId }) {

    const { loading, error, data } = useQuery(MEDIA, {variables: { id: mediaId }});
    if (loading || error) return null

    const mediaElement = data.Media as Media

    console.log(mediaElement)

    return (
        <React.Fragment>
            <div className="card mb-3">
                <div className="row g-0">
                    <div className="col-md-3 p-2">
                        <img src={mediaElement.content.url} alt={mediaElement.content.id} width={200} />
                    </div>
                    <div className="col-md-7">
                        <div className="card-body">
                            <p className="card-text mb-4">{mediaElement.note[0].text}</p>
                            <p className="card-text mb-1"><small className="text-muted">Body site: {mediaElement.bodySite.text}</small></p>
                            <p className="card-text"><small className="text-muted">Status: {mediaElement.status}</small></p>
                        </div>
                    </div>
                    <div className="col-md-2 pt-5">
                        <button className="btn btn-primary">
                            AI Analyze
                        </button>
                    </div>
                </div>
            </div>
        </React.Fragment>
    )
}