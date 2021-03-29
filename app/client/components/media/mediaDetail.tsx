import React from "react"
import { useQuery } from "@apollo/client";

import { MODEL_API } from "../../secrets"

import { Media } from "../../types/fhir"
import { MEDIA } from "../../graphQl/queries/media"

export default function MediaDetail({ mediaId }) {

    const [status, setStatus] = React.useState(null)
    const [isLoading, setLoading] = React.useState(false)

    const { loading, error, data } = useQuery(MEDIA, {variables: { id: mediaId }});
    if (loading || error) return null

    const mediaElement = data.Media as Media

    const analyzeImage = async () => {
        try {
            console.log("sending request ...")
            setLoading(true)
            const res = await fetch(MODEL_API, {
                method: 'POST', // or 'PUT'
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: mediaElement.content.url
                }),
            })
            .then(response => response.json())
            const { prediction } = res
            console.log("Received: ", res)
            setStatus(`Analyzed - Chance of being malignant: ${(parseFloat(prediction) * 100).toPrecision(4)} %`)
            setLoading(false)
        } catch (error) {
            console.error(error)
        }
    }

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
                            <p className="card-text"><small className="text-muted">Status: {status || mediaElement.status}</small></p>
                        </div>
                    </div>
                    <div className="col-md-2 pt-5">
                        <button className="btn btn-primary" onClick={analyzeImage}>
                            {isLoading ? (
                                <div className="spinner-border" role="status">
                                    <span className="sr-only"></span>
                                </div>
                            ) : "Analyze"}
                        </button>
                    </div>
                </div>
            </div>
        </React.Fragment>
    )
}